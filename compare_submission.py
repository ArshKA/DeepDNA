import glob
import re

testing_file = 'predictions6'


def read_submission(file_path):
    with open(file_path) as file:
        sequences = file.readlines()
    sequences = set([x.strip() for x in sequences])
    assert len(sequences) == 10000
    return set(sequences)


print(f'Calculating similarities for {testing_file}')

testing_sequences = read_submission(f'submissions/{testing_file}.csv')

submission_files = glob.glob("submissions/*")
for submission in submission_files:
    file_name = re.match('submissions/([a-zA-Z0-9_.-]+).csv', submission).group(1)
    if file_name == testing_file:
        continue
    submission_sequences = read_submission(submission)

    similarity = len(testing_sequences & submission_sequences)

    print(f'{similarity/10000:.2%} similar with {similarity} common sequences with {file_name}')
