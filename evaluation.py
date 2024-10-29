import torch

def evaluate(model, data_loader):
    """
    Return two dicts: {'PRE': precision, 'RE': recall, 'F1': harmonic mean between Precision and Recall}, {'PRE': precision, 'ACC': accuracy}
    """
    with torch.no_grad():
        num = len(data_loader)
        sgret = {'PRE': 0, 'RE': 0, 'F1': 0}
        srret = {'PRE': 0, 'ACC': 0}
        for data, target in data_loader:
            golden_word, golden_candidates = target
            golden_word = golden_word[0]
            golden_candidates = [c[0][0] for c in golden_candidates]

            word = model.CWI(data[0])
            candidates = model.SG(data[0], word)
            sgpre = 0
            sgre = 0
            for candidate in candidates:
                if candidate in golden_candidates:
                    sgpre += 1
                    sgre += 1
            sgpre /= len(candidates)
            sgre /= len(golden_candidates)
            if sgpre + sgre == 0:
                sgf1 = 0
            else :
                sgf1 = 2 * sgpre * sgre / (sgpre + sgre)
            sgret['PRE'] += sgpre
            sgret['RE'] += sgre
            sgret['F1'] += sgf1

            candidate = model.SR(data[0], word, candidates)
            if candidate == golden_word:
                srret['PRE'] += 1
            if candidate in golden_candidates:
                srret['ACC'] += 1
                srret['PRE'] += 1
    sgret['PRE'] /= num
    sgret['RE'] /= num
    sgret['F1'] /= num
    srret['PRE'] /= num
    srret['ACC'] /= num
    return sgret, srret          