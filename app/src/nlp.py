from app.src import learner_creator


async def nlp_generation(path, data):
    learn = await learner_creator.setup_learner(path)
    entry_text = data['entry_text']
    nb_words = data['nb_words']
    randomness = data['randomness']
    if entry_text:
        return learn.predict(entry_text, int(nb_words), temperature=float(randomness))