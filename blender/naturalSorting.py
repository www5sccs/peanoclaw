def getNaturalsortKey(s):
  "Used internally to get an integer out of the string."
  import re
  matches =  re.findall(r'(\d+)', s)
  if len(matches) != 1:
    raise Exception("Filename should contain one number, indicating the timestep number. Got " + s + ". Found " + str(len(matches)) + " numbers.")

  return int(matches[0])

def naturalSort(seq):
  import re
  return sorted(seq, key=getNaturalsortKey) 


