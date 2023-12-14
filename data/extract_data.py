import math
import pandas as pd
import re

# Define the function to extract book summaries
def extract_book_summaries(file_path, num_samples=100, output_file="extracted_100_summaries.txt"):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().split('\n\n')  # Assuming summaries are separated by double newline

        # Extract 'num_samples' summaries or all available summaries if there are fewer
        samples = data[:num_samples]

        # Write the extracted summaries to a new file
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write('\n\n'.join(samples))


# Function to extract and save book summaries
def extract_summaries(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read().split('\n')  # Assuming summaries are separated by double newline

        count = 0
        summaries = []  # To store extracted summaries

        for block in data:
            if count == 100:
                break
            parts = block.split('\t')
            summary = parts[-1].strip()  # Extracting the last part as the summary
            summaries.append(summary)  # Append the summary to the list
            count += 1

        # Write extracted summaries to a new file
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write('\n\n'.join(summaries))  # Separate summaries by double newline



def get_file(filepath):
    lines=[]
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line!='\n':
                #line_no_periods=[sentence+'.' for sentence in line.split('.')]
                #lines.append([sentence+'.' for sentence in line.split('.')])
                #lines.append([sentence +'.' if sentence!='\n' else sentence for sentence in line.split('.')])
                #print([sentence +'.' if sentence!='\n' else sentence for sentence in line.split('.')])
                lines.append(line.split('.'))
    return lines

def write_file(list_texts, filepath):
    with open(filepath, 'w') as file:
        for line in list_texts:
            file.write('.'.join(line))


def add_rephrases_to_summary(summaries_path, rephrases_path, output):
    summaries=get_file(summaries_path)
    
    df=pd.read_csv(rephrases_path, encoding='latin-1')
    mins, maxs=0, 100
    for i in range(10):
        subset=df.iloc[mins:maxs]
        for index, summary in enumerate(summaries):
            sentence=subset.loc[index+mins, 'Rephrase'].strip('.')
            if len(summary)<=10:
                if i<5:
                    summaries[index]=[sentence]+summary

                else:
                    summaries[index]=summary[:-1]+[sentence]

            else:
                place=math.floor(len(summary)*(mins/1000))
                summary.insert(place, subset.loc[index+mins, 'Rephrase'])
                summaries[index]=summary
        mins+=100
        maxs+=100
    write_file(summaries, output)
    

# file_path = 'booksummaries.txt'
# output_file = "extracted_100_summaries.txt"
# extract_summaries(file_path, output_file)
add_rephrases_to_summary("data/extracted_100_summaries.txt", 'data/10_gold_phrases_and_1000_rephrases.csv', 'data/extracted_100_summaries_with_rephrase.txt')
