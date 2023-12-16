with open('submission_sequences/predictions_seq3.csv', 'r') as file:

    preds = [x.strip() for x in file.readlines()]

with open('raw_data/test.fasta', 'r') as file:
    a = file.readlines()

a = [x.strip() for x in a]

test = {}
for x in a:
    if x[0] == '>':
        cur_seq = x[1:]
        test[cur_seq] = ""
    else:
        test[cur_seq] += x

seq_to_id = {y:x for x, y in test.items()}

with open('submissions/predictions3.csv', 'w') as file:
    for p in preds:
        file.write(seq_to_id[p] + '\n')

print(list(test.items())[0])

print(preds[0] in seq_to_id)