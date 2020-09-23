import pickle


def save(obj, filename, compressed=False):
    """ obj를 pickle 파일(.pkl)로 저장 """
    if compressed:
        import gzip
        with gzip.open(filename, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)


def load(filename, compressed=False):
    """ pickle 파일(.pkl)로부터 로드한 데이터를 반환 """
    if compressed:
        import gzip
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def list_to_dict(values, keys):
    """ key를 value 아이템들의 list로 매핑하는 dict를 반환
     Ex) [item1, item2, item3] => {'key1': [item1, item3], 'key2': [item2]} """
    # convert function to iterable
    if callable(keys):
        keys = map(keys, values)

    root = {}
    for key, value in zip(keys, values):
        nested = root.setdefault(key, [])
        nested.append(value)
    return root
