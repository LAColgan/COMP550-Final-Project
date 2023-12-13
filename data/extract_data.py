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


file_path = 'booksummaries.txt'
output_file = "extracted_100_summaries.txt"
extract_summaries(file_path, output_file)
