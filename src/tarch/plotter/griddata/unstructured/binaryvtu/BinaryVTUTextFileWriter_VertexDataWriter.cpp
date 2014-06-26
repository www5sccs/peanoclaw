#include "tarch/plotter/griddata/unstructured/binaryvtu/BinaryVTUTextFileWriter.h"

#include <limits>
#include <iomanip>

tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter::VertexDataWriter(
  const std::string& dataIdentifier, BinaryVTUTextFileWriter& writer, int recordsPerVertex
):
  _lastWriteCommandVertexNumber(-1),
  _myWriter(writer),
   _dataID(""),
  _vertexDataWriterContentList(),
  _recordsPerVertex(recordsPerVertex),
  _minValue(std::numeric_limits<double>::max()),
  _maxValue(std::numeric_limits<double>::min()) {
  assertion(_recordsPerVertex>0);

  //_out << std::setprecision(_myWriter._precision); precission not necessary anymore?
  if (_recordsPerVertex!=3) {
	  _dataID = dataIdentifier;
#ifdef Parallel
	  _outPVTU << "<PDataArray Name=\"" << dataIdentifier << "\" NumberOfComponents=\"" << _recordsPerVertex << "\" format=\"appended\" type=\"Float32\"/>" << std::endl;
#endif
  }
  else {
	  _dataID = dataIdentifier;
#ifdef Parallel
	  _outPVTU << "<PDataArray Name=\"" << dataIdentifier << "\" NumberOfComponents=\"3\" format=\"appended\" type=\"Float32\"/>" << std::endl;
#endif
  }

  #ifdef Asserts
  _dataIdentifier = dataIdentifier;
  #endif
}


tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter::~VertexDataWriter() {
  if (_lastWriteCommandVertexNumber>=-1) {
    close();
  }
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter::close() {
  assertionEquals2( _lastWriteCommandVertexNumber, _myWriter._numberOfVertices-1, _dataIdentifier, "perhaps not all vertices were assigned data (holds if _myWriter._numberOfVertices is bigger than local attribute)" );
  assertionMsg( _myWriter.isOpen(), "Maybe you forgot to call close() on a data writer before you destroy your writer for value " << _dataIdentifier );

  if (_lastWriteCommandVertexNumber>=-1) {
    _myWriter._numberOfVertexData ++;
    _myWriter._pointDataNumbers.push_back(_recordsPerVertex);
    _myWriter._pointDataIdentifiers.push_back(_dataID);
    _myWriter._pointDataContent.splice(_myWriter._pointDataContent.end(), _vertexDataWriterContentList);
    //_myWriter._pointDataContent.insert(_myWriter._pointDataContent.end(), _vertexDataWriterContentList.begin(), _vertexDataWriterContentList.end());
#ifdef Parallel
    _myWriter._vertexDataDescriptionParallel += _outPVTU.str();
#endif
  }
  _lastWriteCommandVertexNumber = -2;
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter::plotVertex( int index, double value ) {
  assertion(_lastWriteCommandVertexNumber>=-1);
  assertion(1<=_recordsPerVertex);

  assertion2( value != std::numeric_limits<double>::infinity(), index, value);
  assertion2( value == value, index, value);  // test for not a number

  while (_lastWriteCommandVertexNumber<index-1) {
    plotVertex(_lastWriteCommandVertexNumber+1,0.0);
  }

  _lastWriteCommandVertexNumber = index;
   _vertexDataWriterContentList.push_back(float(value));
  for (int i=1; i<_recordsPerVertex; i++) {
     _vertexDataWriterContentList.push_back(0.0);
  }

  if (value<_minValue) _minValue = value;
  if (value>_maxValue) _maxValue = value;
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter::plotVertex( int index, const tarch::la::Vector<2,double>& value ) {
  assertion(_lastWriteCommandVertexNumber>=-1);
  assertion(2<=_recordsPerVertex);

  assertion1( value(0) != std::numeric_limits<double>::infinity(), value(0) );
  assertion1( value(0) == value(0), value(0) );  // test for not a number

  assertion1( value(1) != std::numeric_limits<double>::infinity(), value(1) );
  assertion1( value(1) == value(1), value(1) );  // test for not a number

  while (_lastWriteCommandVertexNumber<index-1) {
    plotVertex(_lastWriteCommandVertexNumber+1,0.0);
  }

  _lastWriteCommandVertexNumber = index;

  _vertexDataWriterContentList.push_back(float(value(0)));
  _vertexDataWriterContentList.push_back(float(value(1)));
  for (int i=2; i<_recordsPerVertex; i++) {
     _vertexDataWriterContentList.push_back(0.0);
  }

  if (value(0)<_minValue) _minValue = value(0);
  if (value(0)>_maxValue) _maxValue = value(0);
  if (value(1)<_minValue) _minValue = value(1);
  if (value(1)>_maxValue) _maxValue = value(1);
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter::plotVertex( int index, const tarch::la::Vector<3,double>& value ) {
  assertion(_lastWriteCommandVertexNumber>=-1);
  assertion(3<=_recordsPerVertex);

  assertion1( value(0) != std::numeric_limits<double>::infinity(), value(0) );
  assertion1( value(0) == value(0), value(0) );  // test for not a number

  assertion1( value(1) != std::numeric_limits<double>::infinity(), value(1) );
  assertion1( value(1) == value(1), value(1) );  // test for not a number

  assertion1( value(2) != std::numeric_limits<double>::infinity(), value(2) );
  assertion1( value(2) == value(2), value(2) );  // test for not a number

  while (_lastWriteCommandVertexNumber<index-1) {
    plotVertex(_lastWriteCommandVertexNumber+1,0.0);
  }

  _lastWriteCommandVertexNumber = index;

  _vertexDataWriterContentList.push_back(float(value(0)));
  _vertexDataWriterContentList.push_back(float(value(1)));
  _vertexDataWriterContentList.push_back(float(value(2)));
  for (int i=3; i<_recordsPerVertex; i++) {
     _vertexDataWriterContentList.push_back(0.0);
  }

  if (value(0)<_minValue) _minValue = value(0);
  if (value(0)>_maxValue) _maxValue = value(0);
  if (value(1)<_minValue) _minValue = value(1);
  if (value(1)>_maxValue) _maxValue = value(1);
  if (value(2)<_minValue) _minValue = value(2);
  if (value(2)>_maxValue) _maxValue = value(2);
}


double tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter::getMinValue() const {
  return _minValue;
}


double tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter::getMaxValue() const {
  return _maxValue;
}
