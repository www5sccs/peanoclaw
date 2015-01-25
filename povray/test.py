from argparse import ArgumentParser
import povray

parser = ArgumentParser()
parser.add_argument('fileNumber', type=int)
arguments = parser.parse_args()

povray.render('../testscenarios/breakingDam/vtkOutput/adaptive-' + str(arguments.fileNumber) + '.hdf5', 'test' + str(arguments.fileNumber) + '.png')
