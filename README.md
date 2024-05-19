# byBS
*A Program for Analyzing the Context and Semantics to Speech*

## Overview
This tool is a sophisticated application designed to transcribe and analyze audio recordings. It features a streamlined tkinter GUI for user interaction and supports .wav audio file format for input. The program is compatible with Jupyter Notebooks, offering a seamless experience for data analysis and machine learning tasks.

## Features

Speech Recognition: Utilizes Sphinx recognizer to transcribe audio files.

BERT Classification: Employs a fine-tuned BERT model for sequence classification to assess the transcription against communication standards.

Criteria-based Analysis: Analyzes the transcription for predefined criteria to identify any communication violations.

Tkinter GUI: Provides a simple and intuitive user interface for selecting and processing audio files.

Jupyter Notebook Support: Ensures compatibility with Jupyter Notebooks for an integrated development and analysis environment.

## Prerequisites

Before running the program, ensure you have the following installed:
Python 3.6 or higher
SpeechRecognition
Transformers
PyTorch
Tkinter

## Installation

To set up the program, follow these steps:
Clone the repository to your local machine.
Install the required Python packages using pip:

pip install -r requirements.txt
Run the program:

python audio_analysis_tool.py

## Usage

Launch the application using the provided script.
Click on the “Выбрать аудиофайлы” button to browse and select a .wav audio file.
Click on the “Анализировать” button to process the selected audio file.
The results of the transcription and analysis will be displayed in the text box within the GUI.

## Jupyter Notebook Integration

To use this tool within a Jupyter Notebook:
Import the necessary functions from the script.
Call the functions with the appropriate parameters within your notebook cells.

## Contributing

Contributions to improve the byBS are welcome. Please follow the standard fork-and-pull request workflow.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
