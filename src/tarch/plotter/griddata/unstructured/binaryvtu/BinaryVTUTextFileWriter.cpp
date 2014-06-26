#include "tarch/plotter/griddata/unstructured/binaryvtu/BinaryVTUTextFileWriter.h"

#include <stdio.h>
#include <fstream>
#include <iomanip>

tarch::logging::Log tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::_log( "tarch::plotter::griddata::unstructured::vtu::VTUTextFileWriter" );


tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::BinaryVTUTextFileWriter(const int precision):
      _writtenToFile(false),
      _precision(precision),
      _doubleOrFloat(setDoubleOrFloatString(precision)),
      _numberOfVertices(0),
      _numberOfCells(0),
      _numberOfCellData(0),
      _numberOfVertexData(0),
      _numberOfCellEntries(0),
      _cellDataContent(),
      _pointDataContent() {}


tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::~BinaryVTUTextFileWriter() {
  if (!_writtenToFile) {
    assertionEqualsMsg( _numberOfVertices,    0, "Still vertices in vtu writer pipeline. Maybe you forgot to call writeToFile() on a data vtu writer?" );
    assertionEqualsMsg( _numberOfCells,       0, "Still cells in vtu writer pipeline. Maybe you forgot to call writeToFile() on a data vtu writer?" );
    assertionEqualsMsg( _numberOfCellEntries, 0, "Still cell entries in vtu writer pipeline. Maybe you forgot to call writeToFile() on a data vtu writer?" );
  }
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::clear() {
  _writtenToFile       = false;
  _numberOfVertices    = 0;
  _numberOfCells       = 0;
  _numberOfCellEntries = 0;
  _numberOfCellData    = 0;
  _numberOfVertexData  = 0;
#ifdef Parallel
  _vertexDataDescriptionParallel  = "";
  _cellDataDescriptionParallel    = "";
#endif
}

int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::calculateOffset(int currentOffset, int elements, int numberOfComponents, int size){
  int offset = currentOffset + sizeof(int) + elements*numberOfComponents*size;
  return offset;
}

int tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::calculateBinaryHeader(int elements, int numberOfComponents, int size){
  int header = elements*numberOfComponents*size;
  return header;
}

void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::appendCellData(FILE * pFile, int * arrayOfCellDataNumbers){
  //write CellData
  for(int i=0; i<_numberOfCellData; i++){
    int dataHeader[1] = {calculateBinaryHeader(_numberOfCells,arrayOfCellDataNumbers[i],sizeof(float))};
    fwrite(dataHeader, sizeof(int), 1, pFile);
    for(int j=0; j<(_numberOfCells*arrayOfCellDataNumbers[i]); j++){
      fwrite(&_cellDataContent.front() , sizeof(float), 1, pFile);
      _cellDataContent.pop_front();
    }
  }
}

void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::appendPointData(FILE * pFile, int * arrayOfPointDataNumbers){
  //write PointData
  for(int i=0; i<_numberOfVertexData; i++){
    int header[1] = {calculateBinaryHeader(_numberOfVertices,arrayOfPointDataNumbers[i],sizeof(float))};
    fwrite(header, sizeof(int), 1, pFile);
    for(int j=0; j<(_numberOfVertices*arrayOfPointDataNumbers[i]); j++){
      fwrite(&_pointDataContent.front(), sizeof(float), 1, pFile);
      _pointDataContent.pop_front();
    }
  }
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::appendCells(FILE * pFile){
  //Write Cells (Connectivity, Offsets, Types)
  int header[1];
  header[0] = calculateBinaryHeader(_sizeOfConnectivity,1,sizeof(int));
  fwrite(header, sizeof(int), 1, pFile);
  for(int i=0; i<_sizeOfConnectivity;i++){
    fwrite(&_cellConnectivity.front(), sizeof(int), 1,pFile);
    _cellConnectivity.pop_front();
  }

  header[0] = calculateBinaryHeader(_numberOfCells,1,sizeof(int));
  fwrite(header, sizeof(int), 1, pFile);
  for(int i=0;i<_numberOfCells; i++){
    fwrite(&_cellOffsets.front(), sizeof(int), 1,pFile);
    _cellOffsets.pop_front();
  }

  header[0] = calculateBinaryHeader(_numberOfCells,1,sizeof(int));
  fwrite(header, sizeof(int), 1, pFile);
  for(int i=0;i<_numberOfCells; i++){
    fwrite(&_cellTypes.front(), sizeof(int), 1,pFile);
    _cellTypes.pop_front();
  }
}

void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::appendPoints(FILE * pFile){
  //Write Points
  int header[1] = {calculateBinaryHeader(_numberOfVertices,3,sizeof(float))};
  fwrite(header, sizeof(int), 1, pFile);
  for(int i=0; i<(_numberOfVertices*3);i++){
    fwrite(&_pointPosition.front(), sizeof(float), 1, pFile);
    _pointPosition.pop_front();
  }
}


#ifdef Parallel
void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::writePVTU(const std::string& filenameComplete, const std::string& filenameShort, int timestepnumber){
  std::ofstream out;
  out.open( filenameComplete.c_str() );
  if ( (!out.fail()) && out.is_open() ) {
    _log.debug( "close()", "opened data file " + filenameComplete );

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>" << std::endl;
    out << "<VTKFile byte_order=\"LittleEndian\" type=\"PUnstructuredGrid\" version=\"0.1\">" << std::endl;
    out << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    if (_numberOfVertices>0 && !_vertexDataDescriptionParallel.empty()) {
      out << "<PPointData>" << std::endl;
      out << _vertexDataDescriptionParallel;
      out << "</PPointData>" << std::endl;
    }

    if (_numberOfCells>0 && !_cellDataDescriptionParallel.empty() ) {
      out << "<PCellData>" << std::endl;
      out << _cellDataDescriptionParallel;
      out << "</PCellData>" << std::endl;
    }


    out << "<PPoints>" << std::endl;
    out << "<PDataArray Name=\"points\" NumberOfComponents=\"3\" format=\"binary\" type=\"Float32\"/>" << std::endl;
    out << "</PPoints>" << std::endl;


    out << "<PCells>" << std::endl;
    out << "<PDataArray Name=\"connectivity\" NumberOfComponents=\"1\" format=\"appended\" type=\"Int32\"/>" << std::endl;
    out << "<PDataArray Name=\"offsets\" NumberOfComponents=\"1\" format=\"appended\" type=\"Int32\"/>" << std::endl;
    out << "<PDataArray Name=\"types\" NumberOfComponents=\"1\" format=\"appended\" type=\"UInt8\"/>" << std::endl;
    out << "</PCells>" << std::endl;

    for(int i=0; i<tarch::parallel::Node::getInstance().getNumberOfNodes(); i++){
      out << "<Piece Source=\"" << filenameShort  << ".cpu-"
          << std::setw(
              (unsigned int) (log(
                  (double) tarch::parallel::Node::getInstance().getNumberOfNodes())
                  / log(10.0)) + 1) << std::setfill('0') << i << ".";
      out << timestepnumber << std::string(".vtu");
      out << "\"/>" << std::endl;
    }
    out << "</PUnstructuredGrid>" << std::endl;
    out << "</VTKFile>" << std::endl;
    _log.debug( "close()", "data written to " + filenameComplete );
  }
  else {
    _log.error( "close()", "unable to write output file " + filenameComplete );
  }

  _writtenToFile = true;

}
#endif



void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::writeToFile( const std::string& filename ) {
  assertion( !_writtenToFile );

  int offset = 0;
  int *arrayOfCellDataNumbers = new int [_numberOfCellData];
  int *arrayOfPointDataNumbers = new int [_numberOfVertexData];
  int arrCellDataCtr = 0;
  int arrPointDataCtr = 0;

  std::ofstream out;
  out.open( filename.c_str() );
  if ( (!out.fail()) && out.is_open() ) {
    _log.debug( "close()", "opened data file " + filename );

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>" << std::endl;
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"<< std::endl;
    out << "<UnstructuredGrid>" << std::endl;
    out << "<Piece NumberOfPoints=\""<< _numberOfVertices << "\" NumberOfCells=\"" << _numberOfCells <<"\">" << std::endl;
    out << "<Points>" << std::endl;
    out << "<DataArray Name=\"points\" type=\"Float32\" NumberOfComponents=\""<< "3"<<"\" format=\"appended\" offset=\"" << offset << "\"/>" << std::endl;
    offset = calculateOffset(offset, _numberOfVertices, 3, sizeof(int));
    out << "</Points>" << std::endl;

    out << "<Cells>" << std::endl;
    out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << offset << "\"/>" << std::endl;
    offset = calculateOffset(offset, _sizeOfConnectivity, 1, sizeof(int));
    out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << offset << "\"/>" << std::endl;
    offset = calculateOffset(offset, _numberOfCells, 1, sizeof(int));
    out << "<DataArray type=\"Int32\" Name=\"types\" format=\"appended\" offset=\"" << offset << "\"/>" << std::endl;
    offset = calculateOffset(offset, _numberOfCells, 1, sizeof(int));
    out << "</Cells>" << std::endl;


    if (_numberOfVertices>0 && !_pointDataContent.empty()) {
      out << "<PointData>" << std::endl;
      for(int i=0; i<_numberOfVertexData;i++){
        out << "<DataArray Name=\"" << _pointDataIdentifiers.front() << "\" NumberOfComponents=\"" << _pointDataNumbers.front() << "\" format=\"appended\" type=\"Float32\" offset=\"" << offset << "\"/>" << std::endl;
        offset = calculateOffset(offset, _numberOfVertices, _pointDataNumbers.front(),sizeof(float));
        _pointDataIdentifiers.pop_front();
        arrayOfPointDataNumbers[arrPointDataCtr++] = _pointDataNumbers.front();
        _pointDataNumbers.pop_front();
      }
      out << "</PointData>" << std::endl;
    }

    if (_numberOfCells>0 && !_cellDataContent.empty() ) {
      out << "<CellData>" << std::endl;
      for(int i=0; i< _numberOfCellData; i++){
        out << "<DataArray Name=\"" << _cellDataIdentifiers.front() << "\" NumberOfComponents=\"" <<_cellDataNumbers.front() <<"\" format=\"appended\" type=\"Float32\" offset=\"" << offset << "\"/>" << std::endl ;
        offset = calculateOffset(offset, _numberOfCells, _cellDataNumbers.front(),sizeof(float));
        _cellDataIdentifiers.pop_front();
        arrayOfCellDataNumbers[arrCellDataCtr++] = _cellDataNumbers.front();
        _cellDataNumbers.pop_front();
      }
      out << "</CellData>" << std::endl;
    }

    out << "</Piece>" << std::endl;
    out << "</UnstructuredGrid>" << std::endl;
    out << "<AppendedData encoding=\"raw\">" << std::endl;
    out << "_";

    //append BINARY DATA start
    out.close();

    FILE * pFile = fopen(filename.c_str() , "ab");
    appendPoints(pFile);
    appendCells(pFile);
    appendPointData(pFile, arrayOfPointDataNumbers);
    appendCellData(pFile, arrayOfCellDataNumbers);
    fclose(pFile);
    //append BINARY DATA stop


    out.open(filename.c_str(), std::ofstream::out | std::ofstream::app);
    out << std::endl << "</AppendedData>" << std::endl;
    out << "</VTKFile>" << std::endl;

    delete[] arrayOfCellDataNumbers;
    delete[] arrayOfPointDataNumbers;

    _log.debug( "close()", "data written to " + filename );

  }
  else {
    _log.error( "close()", "unable to write output file " + filename );
  }

  _writtenToFile = true;
}

bool tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::isOpen() {
  return !_writtenToFile;
}


tarch::plotter::griddata::unstructured::UnstructuredGridWriter::VertexWriter*
tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::createVertexWriter() {
  return new tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexWriter(*this);
}


tarch::plotter::griddata::unstructured::UnstructuredGridWriter::CellWriter*
tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::createCellWriter() {
  return new tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellWriter(*this);
}


void tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::validateDataWriterIdentifier( const std::string& identifier ) const {
  if (identifier.empty()) {
    logWarning(
        "validateDataWriterIdentifier(string)",
        "identifier for vtu file is empty. Spaces are not allowed for vtu data field identifiers and some vtu visualisers might crash."
    );
  }
  if (identifier.find(' ')!=std::string::npos) {
    logWarning(
        "validateDataWriterIdentifier(string)",
        "identifier \"" << identifier << "\" contains spaces. Spaces are not allowed for vtu data field identifiers and some vtu visualisers might crash."
    );
  }
}


tarch::plotter::griddata::Writer::CellDataWriter*    tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::createCellDataWriter( const std::string& identifier, int recordsPerCell ) {
  validateDataWriterIdentifier(identifier);
  return new tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::CellDataWriter(identifier,*this, recordsPerCell);
}


tarch::plotter::griddata::Writer::VertexDataWriter*  tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::createVertexDataWriter( const std::string& identifier, int recordsPerVertex ) {
  validateDataWriterIdentifier(identifier);
  return new tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter::VertexDataWriter(identifier,*this, recordsPerVertex);
}
