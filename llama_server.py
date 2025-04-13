import logging
import json
import numpy as np

import requests
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)

# Helper function similar to the reference implementation
def _get_logprob_of_token(response_data, token_string):
    """Extracts the log probability of a specific token string from a /v1/completions response."""
    try:
        # Ensure structure exists before indexing
        choices = response_data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            logger.warning("Response missing 'choices' list.")
            return -np.inf

        logprobs_obj = choices[0].get("logprobs")
        if not logprobs_obj or not isinstance(logprobs_obj, dict):
            logger.warning("Response choice missing 'logprobs' dictionary.")
            return -np.inf

        logprobs_content = logprobs_obj.get("content")
        if not logprobs_content or not isinstance(logprobs_content, list) or len(logprobs_content) == 0:
             logger.warning("Could not find logprobs 'content' for the predicted token.")
             return -np.inf

        # Expecting content for the single predicted token (max_tokens=1 in loglikelihood)
        first_pos_info = logprobs_content[0]
        if not isinstance(first_pos_info, dict):
             logger.warning("Logprobs 'content' item is not a dictionary.")
             return -np.inf

        first_pos_top_logprobs = first_pos_info.get("top_logprobs")
        if not first_pos_top_logprobs or not isinstance(first_pos_top_logprobs, list):
             logger.warning("Could not find 'top_logprobs' list for the predicted token.")
             return -np.inf

        # Search for the specific token string in the top logprobs list
        for item in first_pos_top_logprobs:
             if isinstance(item, dict) and item.get("token") == token_string:
                  logprob = item.get("logprob")
                  if isinstance(logprob, (int, float)):
                      return float(logprob)
                  else:
                       logger.warning(f"Found token '{token_string}' but logprob is not a number: {logprob}")
                       return -np.inf # Treat invalid logprob as error

        # If the loop completes without finding the token_string
        # This means the actual continuation token was not among the top N predicted tokens.
        # logger.debug(f"Token '{token_string}' not found in top_logprobs: {first_pos_top_logprobs}")
        return -np.inf # Treat as infinitely unlikely for likelihood calculation
    except (IndexError, KeyError, TypeError) as e:
        logger.error(f"Error parsing logprobs response structure: {e}. Response snippet: {str(response_data)[:500]}")
        return -np.inf


@register_model("llama-server")
class LlamaServerLM(LM):
    _DEFAULT_MAX_LENGTH = 4096
    _LOGPROBS_N = 10

    def __init__(self, base_url="http://127.0.0.1:8080", api_key=None, model_id=None, truncate=False, **kwargs):
        """
        Initializes the LlamaServerLM adapter to interact with a llama.cpp server.

        Args:
            base_url (str): The base URL of the llama.cpp server (e.g., "http://127.0.0.1:8080").
            api_key (str, optional): API key for the server, if required. Defaults to None.
            model_id (str, optional): The model identifier to use for requests.
                                     If None, a default 'local-model' is used. Corresponds to 'model' field in API calls.
            truncate (bool): Whether to truncate requests to fit max_length. Default is False.
            **kwargs: Additional arguments (e.g., passed from lm_eval).
        """
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_id = model_id if model_id else "local-model"
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._session.headers.update(headers)

        self._fetched_max_length = self._fetch_max_length()
        self._batch_size = 1
        self.truncate = truncate

        logger.info(f"Initialized LlamaServerLM with base_url={self.base_url}, model_id={self.model_id}, max_length={self._fetched_max_length}, truncate={self.truncate}")

    def _fetch_max_length(self):
        """Attempts to fetch the maximum context length from the server's /props or /v1/models endpoint."""
        max_len = self._DEFAULT_MAX_LENGTH
        # Try /props first (common in llama.cpp server)
        try:
             props_resp = self._session.get(f"{self.base_url}/props", timeout=10)
             props_resp.raise_for_status()
             props_data = props_resp.json()
             fetched_len = props_data.get("default_generation_settings", {}).get("n_ctx")
             if fetched_len and isinstance(fetched_len, int) and fetched_len > 0:
                 max_len = fetched_len
                 logger.info(f"Fetched max_length={max_len} from /props")
                 return max_len
             else:
                  logger.warning("No valid 'n_ctx' found in /props default_generation_settings.")
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
             logger.warning(f"Could not fetch max_length from {self.base_url}/props ({e}). Trying /v1/models...")

        # Fallback to /v1/models (OpenAI compatible standard)
        try:
             models_resp = self._session.get(f"{self.base_url}/v1/models", timeout=10)
             models_resp.raise_for_status()
             models_data = models_resp.json()
             # Assuming the first model listed is the relevant one if multiple exist
             if models_data.get("data") and isinstance(models_data["data"], list) and len(models_data["data"]) > 0:
                  model_info = models_data["data"][0]
                  # Look for context_length or similar fields (names vary)
                  # llama.cpp server often adds a 'meta' field
                  fetched_len = model_info.get("meta", {}).get("n_ctx_train") # Check llama.cpp meta field
                  if fetched_len and isinstance(fetched_len, int) and fetched_len > 0:
                       max_len = fetched_len
                       logger.info(f"Fetched max_length={max_len} from /v1/models meta.n_ctx_train")
                       return max_len
                  else:
                       # Check common OpenAI-like fields if meta not found
                       fetched_len = model_info.get("context_window") or model_info.get("context_length")
                       if fetched_len and isinstance(fetched_len, int) and fetched_len > 0:
                           max_len = fetched_len
                           logger.info(f"Fetched max_length={max_len} from /v1/models context field")
                           return max_len
                       else:
                           logger.warning("No context length field found in /v1/models response data.")
             else:
                  logger.warning("No data found in /v1/models response.")
        except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError, KeyError, ValueError) as e_models:
             logger.warning(f"Could not fetch model properties from /v1/models ({e_models}).")

        logger.warning(f"Using default max_length={max_len}.")
        return max_len

    def _request(self, endpoint, json_payload, request_timeout=180):
        """Sends a POST request to the server with error handling and timeout."""
        url = f"{self.base_url}{endpoint}"
        # Ensure model ID is included in the payload if not already present
        if "model" not in json_payload:
             json_payload["model"] = self.model_id
        logger.debug(f"Sending request to {url}. Payload keys: {list(json_payload.keys())}")
        try:
            response = self._session.post(url, json=json_payload, timeout=request_timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"API request timed out ({request_timeout}s) to {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed to {url}: {e}")
            # Attempt to log error response from server if possible
            if e.response is not None:
                 try:
                      err_body = e.response.json()
                      logger.error(f"Server error response ({e.response.status_code}): {err_body}")
                 except json.JSONDecodeError:
                      logger.error(f"Server error response ({e.response.status_code}, non-JSON): {e.response.text[:500]}...")
            return None
        except json.JSONDecodeError as e_json:
            # This can happen if the server returns HTML (e.g., 404 page) or malformed JSON
            logger.error(f"Failed to decode JSON response from {url}: {e_json}")
            # Check if response object exists to log details
            if 'response' in locals() and response is not None:
                logger.error(f"Response status code: {response.status_code}")
                logger.error(f"Response text: {response.text[:500]}...")
            return None

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm, desc="loglikelihood (server)"
        ):
            # Ensure inputs are strings
            context = context if isinstance(context, str) else context.decode('utf-8', errors='ignore')
            continuation = continuation if isinstance(continuation, str) else continuation.decode('utf-8', errors='ignore')

            # Truncate context if needed (leave space for 1 token for the server to predict)
            if self.truncate:
                max_len_context = self.max_length - 1 # Server needs to predict at least one token
                if len(context) > max_len_context:
                    context = context[-max_len_context:]
                    logger.debug(f"Truncated context for loglikelihood to {max_len_context} chars.")

            payload = {
                "prompt": context,       # Send only the context
                "max_tokens": 1,         # We only need the server to predict one token
                "logprobs": self._LOGPROBS_N, # Request top N logprobs to find the continuation's prob
                "temperature": 0.0,      # Make prediction deterministic (though we discard the result)
                # "model": self.model_id # Handled by _request
            }

            response = self._request("/v1/completions", payload)

            if response:
                # Find the log probability of the *actual* continuation token within the response
                logprob_of_continuation = _get_logprob_of_token(response, continuation)
                # is_greedy doesn't strictly matter for loglikelihood comparisons but is required by interface
                is_greedy = True # Assume greedy for comparison tasks like MMLU
                res.append((logprob_of_continuation, is_greedy))
            else:
                logger.error(f"Loglikelihood request failed for context: '{context[:50]}...'")
                res.append((-np.inf, False)) # Indicate failure

        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []
        for request_obj in tqdm(requests, disable=disable_tqdm, desc="generate_until (server)"):
            # request_obj is lm_eval.api.request.Request
            context, gen_kwargs = request_obj.args

            # Extract generation parameters
            stop_sequences = gen_kwargs.get("until", None) # API uses 'stop'
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.8) # Common default
            top_p = gen_kwargs.get("top_p", 0.95) # Common default
            # top_k = gen_kwargs.get("top_k", 40) # Add if server supports it

            # Ensure context is string
            context = context if isinstance(context, str) else context.decode('utf-8', errors='ignore')

            # Apply truncation if enabled
            if self.truncate:
                 max_context_len = self.max_length - max_gen_toks
                 if max_context_len <= 0:
                      logger.warning(f"max_gen_toks ({max_gen_toks}) >= max_length ({self.max_length}). Setting context to empty.")
                      context = ""
                 elif len(context) > max_context_len:
                      context = context[-max_context_len:]
                      logger.debug(f"Truncated context for generate_until to {max_context_len} chars.")

            # Format for Chat API - more standard for generation tasks
            messages = [{"role": "user", "content": context}]

            payload = {
                "messages": messages,
                "max_tokens": max_gen_toks,
                "temperature": temperature,
                "top_p": top_p,
                # "top_k": top_k,
                # "stream": False, # Ensure we get the full response
                # "model": self.model_id # Handled by _request
            }
            # Add stop sequences if provided
            if stop_sequences:
                 payload["stop"] = stop_sequences

            response = self._request("/v1/chat/completions", payload)

            generated_text = "" # Default to empty string
            if response and response.get("choices") and isinstance(response["choices"], list) and len(response["choices"]) > 0:
                 message = response["choices"][0].get("message")
                 if message and isinstance(message, dict):
                     generated_text = message.get("content", "").strip()
                 else:
                     logger.warning(f"Chat completion response missing expected message structure: {response}")

            elif response is None:
                 logger.error(f"generate_until received None response (request failed) for context: '{context[:50]}...'")
            else:
                 logger.error(
                    f"Invalid response structure for generate_until. Context: '{context[:50]}...'. Response: {response}"
                 )

            res.append(generated_text)

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
         # Not easily implemented via standard server endpoints
         raise NotImplementedError("loglikelihood_rolling not implemented for LlamaServerLM")

    # --- LM Properties ---

    @property
    def batch_size(self):
         # Server handles requests individually from our script's perspective
         return self._batch_size

    @property
    def device(self):
         # Computation happens on the server
         return "server"

    @property
    def eot_token_id(self):
        # Cannot reliably get this from the server via standard API.
        # Attempt to tokenize EOS string, though this might vary by model
        try:
            eos_tokens = self.tok_encode("</s>", add_special_tokens=False)
            if eos_tokens and len(eos_tokens) == 1:
                return eos_tokens[0]
        except Exception as e:
            logger.warning(f"Could not determine eot_token_id via tokenization: {e}")
        logger.warning("Returning None for eot_token_id.")
        return None

    @property
    def max_length(self):
        # Return the context length fetched or defaulted
        return self._fetched_max_length

    @property
    def max_gen_toks(self):
        # Default max generation tokens - allow space for context
        # Ensure it's at least a reasonable minimum like 256
        return max(256, self.max_length // 2)

    # --- Tokenization Methods ---

    def tok_encode(self, string: str, add_special_tokens=False):
        # Use the server's /tokenize endpoint if available
        payload = {"content": string, "add_special": add_special_tokens}
        # Add timeout for tokenization as it might hang on very long strings
        response = self._request("/tokenize", payload, request_timeout=60)
        if response and "tokens" in response and isinstance(response["tokens"], list):
            try:
                return [int(t) for t in response["tokens"]]
            except (ValueError, TypeError) as e:
                 logger.error(f"Failed to convert tokens to integers: {response['tokens']}. Error: {e}")
                 raise RuntimeError(f"Server tokenization returned non-integer tokens. Response: {response}") from e
        else:
            logger.error(f"Failed to tokenize string via server: '{string[:50]}...' Response: {response}")
            # Fallback or error? lm-eval needs this.
            raise RuntimeError(f"Server tokenization failed. Endpoint '/tokenize' might be unavailable or returned an error. Response: {response}")


    def tok_decode(self, tokens):
        # Use the server's /detokenize endpoint if available
        # Ensure tokens are standard list of ints
        try:
             tokens_list = [int(t) for t in tokens]
        except (ValueError, TypeError) as e:
             logger.error(f"Input tokens for detokenization are not valid integers: {tokens[:20]}... Error: {e}")
             raise ValueError("Invalid token type for detokenization.") from e

        payload = {"tokens": tokens_list}
        response = self._request("/detokenize", payload, request_timeout=60)
        if response and "content" in response:
            return response["content"]
        else:
            logger.error(f"Failed to detokenize tokens via server: {tokens_list[:10]}... Response: {response}")
            # Fallback or error?
            raise RuntimeError(f"Server detokenization failed. Endpoint '/detokenize' might be unavailable or returned an error. Response: {response}")
