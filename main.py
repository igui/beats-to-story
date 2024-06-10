from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from openai import OpenAI
from typing import Iterable, Optional
import json
from nltk.tokenize import word_tokenize
from contextlib import asynccontextmanager
from typing import NamedTuple
from fastapi.responses import StreamingResponse
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field


# Use your own key
OPENAI_KEY = "change-me"

# Limits on the generated prose.
MIN_WORD_BEAT = 100
MAX_WORD_BEAT = 150

# Whether we generate debug messages
DEBUG = True

# Avoid infinite loops (and spending lots of mony) modifying text by limiting
# the amount of times we modify the string 
MAX_TRIES_MODIFY = 3

class MLModels:
    client: Optional[OpenAI] = None

    def __init__(self):
        self.client = None
    
    def initialize(self):
        debug("Initializing OpenAI")
        self.client = OpenAI(api_key=OPENAI_KEY)
        debug("Downloading nltk data..")
        nltk.download('punkt')
        debug("Initialized!")

    def completions_create(self, **kwargs):
        if self.client is None:
            self.initialize()
        return self.client.chat.completions.create(**kwargs)

ml_models = MLModels()


def chat_with_gpt(prompt: str) -> str:
  completion = ml_models.completions_create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
  )
  return completion.choices[0].message.content


def debug(s: str):
  if DEBUG:
    print(s)


def generate_prose_iter(beat: str, previous_context: list[str]) -> Iterable[str]:
  """
  It will generate prose based on an iterable, based on some previous context
  to make the generated prose follow the story. It generates tokens one by
  one, suitable for streaming
  """

  average_words = int(0.5 * (MIN_WORD_BEAT + MAX_WORD_BEAT))
  messages = [
        # We force a number to give the LLM some leeway but not overgenerate
        # Or undergenerate; the LLM doesn't respect limits in general
        {
          "role": "system", 
          "content": f"You generate prose of around {average_words} words."
        }
  ]

  messages += [
      { "role": "system", "content": previous } for previous in previous_context
  ]

  msg = (f"Generate {average_words} of text of a story based on "
    f"this beat: {beat}")
  messages.append({ "role": "user", "content": msg })

  debug(messages)

  stream = ml_models.completions_create(
    model="gpt-3.5-turbo",
    messages=messages,
    stream=True,
  )

  for chunk in stream:
    if len(chunk.choices) < 1:
      # This shouldn't ever happen, but if it does, we know
      raise ValueError("No choices!")
    choice = chunk.choices[0]
    if choice.delta.content is None:
      continue
    yield choice.delta.content


def alter_string(s: str, how: str) -> str:
  """A simple function to alter a string"""
  response = ml_models.completions_create(
    model="gpt-3.5-turbo",
    response_format={ "type": "json_object" },
    messages=[
        { 
          "role": "user", 
          "content": f"Generate a {how} version of this {s}. Respond in JSON in"
            ' format { "response": "aaa" }'
      }
    ]
  )
  json_response = json.loads(response.choices[0].message.content)
  return json_response['response']


def generate_prose(beat: str, previous_context: list[str]) -> str:
  """
  This is a similar version of generate_prose_iter, but this time
  we make sure the length of the generated string falls between the limits
  of the word count
  """
  result = ''.join(generate_prose_iter(beat, previous_context))

  for _tries in range(MAX_TRIES_MODIFY):
    n_words = len(word_tokenize(result))

    if n_words < MIN_WORD_BEAT:
      debug(f"Warning: Generated too few ({n_words}) words!")
      result = alter_string(result, "longer")
    elif n_words > MAX_WORD_BEAT:
      debug(f"Warning: Generated too many ({n_words}) words!")
      result = alter_string(result, "shorter")
    else:
      return result
  
  debug(f"Max tries reached to alter string")
  return result


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ml_models.initialize
    yield

class GenerationParameters(NamedTuple):
  characters: list[str]
  setting: str
  writing_style: str

  def to_context(self) -> list[str]:
    return [
        f"The characters of this story are: {', '.join(self.characters)}",
        f"The story is set on {self.setting}",
        f"The writing style is {self.writing_style}"
    ]

def generate_story_with_parameters(
    beats: Iterable[str], 
    parameters: GenerationParameters
) -> Iterable[str]:
  """Generate a story based on beats. It will generate it beat by beat"""

  proses = []
  for beat in beats:
    debug(f"Generating prose for beat: {beat}")
    # Don't put so much text (it is expensive to pass many tokens to the GPT :)
    context = parameters.to_context() + proses[-3:]
    generated = generate_prose(beat, context)
    debug(f"Generated prose: {generated}")
    proses.append(generated)
    yield generated

app = FastAPI(lifespan=lifespan)

class BeatsToStoryParams(BaseModel):
    beats: list[str] = Field(title="The beats of the story")
    characters: list[str] = Field(title="The characters of the story")
    setting: str = Field(title="The setting of the story")
    writing_style: str = Field(title="The writing style of the story")


def wrap_generated_beat(proses: Iterable[str]) -> Iterable[str]:
    """Wrap the generated prose in a beat"""
    for prose in proses:
        event = json.dumps({ "type": "generated-beat", "content": prose}) 
        yield f'event: generated-beat\ndata: {event}\n\n'

    yield f'event: end\ndata: {json.dumps({ "type": "end" })}\n\n'

@app.post("/")
async def root(params: BeatsToStoryParams):
    generation_parameters = GenerationParameters(
        characters=params.characters,
        setting=params.setting,
        writing_style=params.writing_style
    )
    proses = generate_story_with_parameters(params.beats, generation_parameters)
    wrapped_generator = wrap_generated_beat(proses)
    return StreamingResponse(wrapped_generator, media_type='text/event-stream')