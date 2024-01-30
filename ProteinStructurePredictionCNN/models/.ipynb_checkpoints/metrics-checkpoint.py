def Q8_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]):  # per amino acid residue
            if np.sum(real[i, j, :]) == 0:  # if it is padding
                total = total - 1
            else:
                if real[i, j] == torch.argmax(pred[i, j]):
                    correct = correct + 1

    return correct / total
