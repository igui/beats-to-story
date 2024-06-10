## Beat to Story

A FastAPI application leverages the OpenAI GPT-3.5 model to generate story prose based on 
user-defined beats, characters, settings, and writing styles.

## Requirements

- Python 3.9
- [FastAPI](https://fastapi.tiangolo.com/)
- [NLTK](https://www.nltk.org/)

## How to run

### Init the server

```bash
poetry install
fastapi dev main.py
```

### Create a sample payload

You can use `sample_payload.json` or create a similar file to set the story beats characters and so 
on.

```json
{
    "beats": [
        "Begin the chapter with Jack and Xander continuing their excavation on the lunar surface, creating a sense of tension and anticipation.",
        "Describe the barren landscape of the moon, emphasizing its desolation and the isolation felt by Jack and Xander.",
        "End the chapter with a cliffhanger or unresolved tension, leaving the reader eager to continue reading and discover what happens next."
    ],
    "characters": [
        "Xander: a rugged, unscropolous miner in his late forties, who wants to be retire soon and return Earth",
        "Jack: a craven but meticulous explorer, driven by curiosity to unlock the mysteries of the cosmos",
    ],
    "setting": "a desolate, barren lunar surface in 2080.",
    "writing_style": "1950's"
}
```

### Invoke the tool
