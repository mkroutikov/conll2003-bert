from ..batcher import Batcher


def test_smoke():

    data = [
        {'words': [1, 2, 3, 4], 'labels': [5, 6, 7, 8]},
        {'words': [2, 2, 1], 'labels': [1, 1, 1]},
    ]

    batched_data = Batcher(data, batch_size=2)

    batched_data = list(batched_data)

    assert len(batched_data) == 1
    words, labels = batched_data[0]

    assert words[0].tolist() == [1, 2, 3, 4]
    assert words[1].tolist() == [2, 2, 1, 0]

    assert labels[0].tolist() == [5, 6, 7, 8]
    assert labels[1].tolist() == [1, 1, 1, 0]
