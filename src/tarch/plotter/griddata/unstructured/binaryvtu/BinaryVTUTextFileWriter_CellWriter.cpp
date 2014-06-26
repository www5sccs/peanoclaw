#include "tarch/plotter/griddata/unstructured/binaryvtu/BinaryVTUTextFileWriter.h"


tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter::CellWriter(BinaryVTUTextFileWriter& writer):
  _currentCellNumber(0),
  _myWriter(writer),
  _cellListEntries(0),
  _offset(0){
  assertion( _myWriter._numberOfCells==0 );
  assertion( _myWriter._numberOfCellEntries==0 );
}


tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter::~CellWriter() {
  if (_currentCellNumber>=0) {
    close();
  }
}


int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter::plotPoint(int vertexIndex) {
  assertion( _currentCellNumber>=0 );
  assertion( _cellListEntries>=0 );

  _currentCellNumber++;
  _cellListEntries += 2;


  _offset += 1;

  _myWriter._cellConnectivity.push_back(vertexIndex);
  _myWriter._cellOffsets.push_back(_offset);
  _myWriter._cellTypes.push_back(1);

  return _currentCellNumber-1;
}


int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter::plotHexahedron(int vertexIndex[8]) {
  assertion( _currentCellNumber>=0 );
  assertion( _cellListEntries>=0 );

  _currentCellNumber++;
  _cellListEntries += 9;


  _offset += 8;

  _myWriter._cellConnectivity.push_back(float(vertexIndex[0]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[1]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[2]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[3]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[4]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[5]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[6]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[7]));
  _myWriter._cellOffsets.push_back(_offset);
  _myWriter._cellTypes.push_back(11);

  return _currentCellNumber-1;
}


int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter::plotQuadrangle(int vertexIndex[4]) {
  assertion( _currentCellNumber>=0 );
  assertion( _cellListEntries>=0 );

  _currentCellNumber++;
  _cellListEntries += 5;


  _offset += 4;

  _myWriter._cellConnectivity.push_back(float(vertexIndex[0]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[1]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[2]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[3]));
  _myWriter._cellOffsets.push_back(_offset);
  _myWriter._cellTypes.push_back(8);


  return _currentCellNumber-1;
}


int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter::plotLine(int vertexIndex[2]) {
  assertion( _currentCellNumber>=0 );
  assertion( _cellListEntries>=0 );

  _currentCellNumber++;
  _cellListEntries += 3;

  _offset += 2;

  _myWriter._cellConnectivity.push_back(float(vertexIndex[0]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[1]));
  _myWriter._cellOffsets.push_back(_offset);
  _myWriter._cellTypes.push_back(3);

  return _currentCellNumber-1;
}


int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter::plotTriangle(int vertexIndex[3]) {
  assertion( _currentCellNumber>=0 );
  assertion( _cellListEntries>=0 );

  _currentCellNumber++;
  _cellListEntries += 4;

  _offset += 3;

  _myWriter._cellConnectivity.push_back(float(vertexIndex[0]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[1]));
  _myWriter._cellConnectivity.push_back(float(vertexIndex[2]));
  _myWriter._cellOffsets.push_back(_offset);
  _myWriter._cellTypes.push_back(5);

  return _currentCellNumber-1;
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter::close() {
  assertion( _myWriter._numberOfCells==0 );
  assertion( _myWriter._numberOfCellEntries==0 );
  assertionMsg( _myWriter.isOpen(), "Maybe you forgot to call close() on a data writer before you destroy your writer?" );

  _myWriter._numberOfCells       = _currentCellNumber;
  _myWriter._numberOfCellEntries = _cellListEntries;

  _myWriter._sizeOfConnectivity = _offset;
  _currentCellNumber = -1;
  _cellListEntries   = -1;


}
