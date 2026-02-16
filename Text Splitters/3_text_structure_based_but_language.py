from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader

text ="""class Student:
    def __init__(self, roll, name, marks):
        self.roll = roll
        self.name = name
        self.marks = marks

    def average(self):
        return sum(self.marks) / len(self.marks)

    def grade(self):
        avg = self.average()
        if avg >= 90: return "A"
        elif avg >= 75: return "B"
        elif avg >= 60: return "C"
        else: return "D"

    def display(self):
        print(f"\nRoll: {self.roll}")
        print(f"Name: {self.name}")
        print(f"Marks: {self.marks}")
        print(f"Average: {self.average():.2f}")
        print(f"Grade: {self.grade()}")

students = []

while True:
    roll = input("\nEnter Roll (or 'exit'): ")
    if roll.lower() == "exit": break
    name = input("Enter Name: ")
    marks = list(map(int, input("Enter marks (space separated): ").split()))
    s = Student(roll, name, marks)
    students.append(s)
    s.display()
"""


splitter = RecursiveCharacterTextSplitter.from_language(
  language=Language.PYTHON,
  chunk_size = 300,
  chunk_overlap = 0,
)

chunk = (splitter.split_text(text))

print(len(chunk))
print(chunk[0])