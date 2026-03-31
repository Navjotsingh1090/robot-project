def normalize_name(raw_name):
    if raw_name is None:
        return None

    words = str(raw_name).strip().split()
    if not words:
        return None

    filler_words = {
        "my", "name", "is", "i", "am", "i'm", "this", "its", "it's",
        "it", "me", "called", "call",
    }

    cleaned_words = []
    for word in words:
        letters_only = "".join(c for c in word if c.isalpha())
        if not letters_only:
            continue
        if letters_only.lower() in filler_words:
            continue
        cleaned_words.append(letters_only)

    if not cleaned_words:
        return None

    name = cleaned_words[0]
    if len(name) < 2:
        return None
    return name.capitalize()
