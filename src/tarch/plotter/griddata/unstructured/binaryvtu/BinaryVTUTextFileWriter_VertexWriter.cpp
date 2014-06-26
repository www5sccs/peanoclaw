#include "tarch/plotter/griddata/unstructured/binaryvtu/BinaryVTUTextFileWriter.h"

#include <iomanip>

tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexWriter::VertexWriter(BinaryVTUTextFileWriter& writer):
  _currentVertexNumber(0),
  _myWriter(writer) {
  assertion( _myWriter._numberOfVertices==0 );
  //_out << std::setprecision(_myWriter._precision); precision not needed anymore?
}


tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexWriter::~VertexWriter() {
  if (_currentVertexNumber>=0) {
    close();
  }
}



int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexWriter::plotVertex(const tarch::la::Vector<2,double>& position) {
  assertion1( _currentVertexNumber>=0, _currentVertexNumber );

  tarch::la::Vector<3,double> p;
  p(0) = position(0);
  p(1) = position(1);
  p(2) = 0.0;

  return plotVertex(p);
}


int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexWriter::plotVertex(const tarch::la::Vector<3,double>& position) {
  assertion( _currentVertexNumber>=0 );
  _currentVertexNumber++;

  _myWriter._pointPosition.push_back(float(position(0)));
  _myWriter._pointPosition.push_back(float(position(1)));
  _myWriter._pointPosition.push_back(float(position(2)));

  return _currentVertexNumber-1;
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexWriter::close() {
  assertion( _myWriter._numberOfVertices==0 );
  assertionMsg( _myWriter.isOpen(), "Maybe you forgot to call close() on a data writer before you destroy your writer?" );


  _myWriter._numberOfVertices  = _currentVertexNumber;
  _currentVertexNumber         = -1;
}
