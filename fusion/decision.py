def decide(probs, th=0.5):
    best = max(probs, key=probs.get)
    return best if probs[best] >= th else None
