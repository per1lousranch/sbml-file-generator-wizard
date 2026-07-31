from libsbml import *
import libsbml
import sys

with open("/Users/zhue3/Documents/GitHub/sbml-file-generator-wizard/noerrors.xml", "r", encoding="utf-8") as file:
    xml_string = file.read()

reader = SBMLReader() # initialize SBMLReader

doc = reader.readSBMLFromString(xml_string) # read from string in output

#print("Internal consistency: " + str(doc.checkInternalConsistency()))
#print("Consistency: " + str(doc.checkConsistency()))

stream = libsbml.ostringstream()
doc.printErrors(stream)
output = stream.str()

#print(output)

