from pydantic import BaseModel,Field
from typing import Optional


class Student(BaseModel):
  name :str = 'Harkirat'  
  age : Optional[int] = None
  cgpa:float=Field(...,gt=0,lt=10)

  
new_student = Student(
    name="Amit",
    age=22,
    cgpa=8.5
)
student_dict =dict((new_student))
print(student_dict['age'])
student_json = new_student.model_dump_json()
print(student_json)