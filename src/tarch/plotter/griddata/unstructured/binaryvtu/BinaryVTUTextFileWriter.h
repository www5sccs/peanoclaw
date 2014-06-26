// Copyright (C) 2009 Technische Universitaet Muenchen
// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www5.in.tum.de/peano
#ifndef _TARCH_PLOTTER_GRIDDATA_UNSTRUCTURED_BINARYVTU_BINARYVTUTEXTFILEWRITER_H_
#define _TARCH_PLOTTER_GRIDDATA_UNSTRUCTURED_BINARYVTU_BINARYVTUTEXTFILEWRITER_H_

#include <list>
#include "tarch/logging/Log.h"
#include "tarch/plotter/griddata/unstructured/UnstructuredGridWriter.h"


namespace tarch {
  namespace plotter {
    namespace griddata {
      namespace unstructured {
        namespace binaryvtu {
          class BinaryVTUTextFileWriter;
        }
      }
    }
  }
}


/**
 * BinaryVTU Writer
 *
 * Output for binaryvtu files (paraview) as text files.
 *
 * !! Usage
 *
 * - Create an instance of the BinaryVTUTextFileWriter.
 * - For the vertices you want to write, you have to create your own
 *   VertexWriter.
 * - Pass all the vertices to this writer (both hanging and persistent nodes).
 *   For each vertex you receive an unique number. Remember this number.
 * - For the elements you want to write, you have to create your own
 *   ElementWriter.
 * - For each record create a data writer. There's two writers: One for the
 *   vertices and one for the cells.
 *
 * !!! Thread-safety
 *
 * The plotter is not thread-safe and shall never be thread-safe: It is the
 * responsibility of the using system to implement thread-safe calls. For
 * Peano, this is the mappings where some events could occur simultaneously.
 *
 * @author Robert Guder, Michael Lieb
 */
class tarch::plotter::griddata::unstructured::binaryvtu::BinaryVTUTextFileWriter:
public tarch::plotter::griddata::unstructured::UnstructuredGridWriter {
private:
  /**
   * Logging device.
   */
  static tarch::logging::Log _log;

  static const std::string HEADER;

  bool _writtenToFile;

  /** precision */
  const int _precision;

  /** either "float" or "double" depending on _precision */
  const std::string _doubleOrFloat;

  /**
   * Total number of vertices
   */
  int _numberOfVertices;

  /**
   * Total number of cells
   */
  int _numberOfCells;

  /**
   * Total number of cell entries. See _cellListEntries.
   */
  int _numberOfCellEntries;


#ifdef Parallel
  std::string _vertexDataDescriptionParallel;
  std::string _cellDataDescriptionParallel;
#endif

  /**
   * List with names of the cell data entries
   */
  std::list<std::string> _cellDataIdentifiers;
  /**
   * "NumberOfComponents" of each DataArray in <CellData>
   */
  std::list<int> _cellDataNumbers;
  /**
   * Complete Data of all the children of "CellData" in the XML File in a single list
   */
  std::list<float> _cellDataContent;

  /**
   * List with names of the PointData entries
   */
  std::list<std::string> _pointDataIdentifiers;

  /**
   * "NumberOfComponents" of each DataArray in <PointData>
   */
  std::list<int> _pointDataNumbers;

  /**
   * Complete Data of all the children of "PointData" in the XML File in a single list
   */
  std::list<float> _pointDataContent;

  /**
   * Complete Data of Cell Connectivity
   */
  std::list<int> _cellConnectivity;

  /**
   * Complete Data of Cell Offsets
   *
   */
  std::list<int> _cellOffsets;

  /**
   * Complete Data of Cell Types
   */
  std::list<int> _cellTypes;

  /**
   * Complete DataArray Values of XML Tag <Points>
   */
  std::list<float> _pointPosition;

  /**
   * Total number of children of "CellData" in the XML File
   */
  int _numberOfCellData;

  /**
   * Total number of children of "PointData" in the XML File
   */
  int _numberOfVertexData;

  /**
   * The sizes of the Data written out in binary format in bytes - necessary for header
   */
  int _sizeOfConnectivity;
  void validateDataWriterIdentifier( const std::string& identifier ) const;

  /*
   * This method is used to append the appropriate data at the end of the vtu file in binary format
   */
  void appendPoints(FILE * pFile);

  /*
   * This method is used to append the appropriate data at the end of the vtu file in binary format
   */
  void appendCells(FILE * pFile);

  /*
   * This method is used to append the appropriate data at the end of the vtu file in binary format
   */
  void appendCellData(FILE * pFile, int * arrayOfCellDataNumbers);

  /*
   * This method is used to append the appropriate data at the end of the vtu file in binary format
   */
  void appendPointData(FILE * pFile, int * arrayOfPointDataNumbers);

  std::string setDoubleOrFloatString(const int precision){
    if (precision < 7){
      return "float";
    } else {
      return "double";
    }
  }


public:
  BinaryVTUTextFileWriter(const int precision=6);
  virtual ~BinaryVTUTextFileWriter();

  virtual bool isOpen();

  /**
   * Calculates the offset. therefore, the last offset must be given as parameter. Furthermore numberOfVertices/numberOfCells is
   * the value of elements. Finally the last paraemter is _numberOfComponents.
   *
   * The Offset is calculated using the last offset and adding the number of entries multiplied with the size of data (4 Bytes for Float32 for example).
   * Additionally 4 Bytes have to be added, which is the header with length-information of the appended data
   */
  virtual int calculateOffset(int currentOffset, int elements, int numberOfComponents, int size);

  /**
   * Calculates the Prefix (Header) of the binary data. It presents the length of the following data
   *
   */
  virtual int calculateBinaryHeader(int elements, int numberOfComponents, int size);

  virtual void writeToFile( const std::string& filename );


#ifdef Parallel
  virtual void writePVTU(const std::string& filenameComplete, const std::string& filenameShort, int timestepnumber);
#endif

  virtual void clear();

  virtual VertexWriter*      createVertexWriter();
  virtual CellWriter*        createCellWriter();
  virtual CellDataWriter*    createCellDataWriter( const std::string& identifier, int recordsPerCell );
  virtual VertexDataWriter*  createVertexDataWriter( const std::string& identifier, int recordsPerVertex );

  /**
   * This is the vertex writer you have to create to plot the vertices.
   * Besides the pure syntax management, the writer also provides a number
   * generator which provides you with a unique id for each vertex.
   *
   * Please ensure that you call close() on the vertex writer before you
   * close the underlying VTUTextFileWriter.
   */
  class VertexWriter:
public tarch::plotter::griddata::unstructured::UnstructuredGridWriter::VertexWriter {
private:
  /**
   * The father class is a friend. There are no other friends.
   */
  friend class BinaryVTUTextFileWriter;

  /**
   * Counter for the vertices written. Holds the maximum index.
   */
  int _currentVertexNumber;

  /**
   * Underlying VTU writer.
   */
  BinaryVTUTextFileWriter& _myWriter;


  VertexWriter(BinaryVTUTextFileWriter& writer);
public:
  virtual ~VertexWriter();

  virtual int plotVertex(const tarch::la::Vector<2,double>& position);
  virtual int plotVertex(const tarch::la::Vector<3,double>& position);

  virtual void close();
};

/**
 * Writes the element data.
 */
class CellWriter:
public tarch::plotter::griddata::unstructured::UnstructuredGridWriter::CellWriter {
private:
  /**
   * The father class is a friend. There are no other friends.
   */
  friend class BinaryVTUTextFileWriter;

  /**
   * Counter for the elements written. Holds the maximum index.
   */
  int _currentCellNumber;

  /**
   * Underlying VTU writer.
   */
  BinaryVTUTextFileWriter& _myWriter;

  /**
   * The tag CELLS in a vtu file requires the number of total entries in the
   * following list of cell-interconnection (for a triangle, an entry could
   * look like this: "3 1 2 4", which states that the triangle has 3 links to
   * vetrices with indices 1, 2 and 4. this makes up four entries). For an
   * unstructured mesh, the element type is not fixed and, hence, the total
   * amount of list entries must be counted by summing up the contributions
   * of each element, when adding the element.
   */
  int _cellListEntries;


  int _offset;



  CellWriter(BinaryVTUTextFileWriter& writer);
public:
  virtual ~CellWriter();

  virtual int plotHexahedron(int vertexIndex[8]);

  virtual int plotQuadrangle(int vertexIndex[4]);

  virtual int plotLine(int vertexIndex[2]);

  virtual int plotTriangle(int vertexIndex[3]);

  virtual int plotPoint(int vertexIndex);

  virtual void close();
};

class CellDataWriter:
public tarch::plotter::griddata::Writer::CellDataWriter {
private:
  /**
   * The father class is a friend. There are no other friends.
   */
  friend class BinaryVTUTextFileWriter;

  /**
   *
   */
  int _lastWriteCommandCellNumber;

  /**
   * Underlying VTU writer.
   */
  BinaryVTUTextFileWriter& _myWriter;

  /**
   * local Copy of _dataIdentifier
   */
  std::string _dataID;

  /**
   * local Copy of the COntent of the single Cells
   */
  std::list<float> _cellDataWriterContentList;
#ifdef Parallel
  /**
   * Output stream
   */
  std::ostringstream _outPVTU;

#endif
  int _recordsPerCell;

  double _minValue;
  double _maxValue;
  CellDataWriter(const std::string& dataIdentifier, BinaryVTUTextFileWriter& writer, int recordsPerCell);
public:
  virtual ~CellDataWriter();

  virtual void close();

  virtual void plotCell( int index, double value );
  virtual void plotCell( int index, const tarch::la::Vector<2,double>& value );
  virtual void plotCell( int index, const tarch::la::Vector<3,double>& value );

  virtual double getMinValue() const;
  virtual double getMaxValue() const;

  void assignRemainingCellsDefaultValues() {}
};

class VertexDataWriter:
public tarch::plotter::griddata::Writer::VertexDataWriter {
private:
  /**
   * The father class is a friend. There are no other friends.
   */
  friend class BinaryVTUTextFileWriter;

  /**
   *
   */
  int _lastWriteCommandVertexNumber;

  /**
   * Underlying VTU writer.
   */
  BinaryVTUTextFileWriter& _myWriter;

  /**
   * local Copy of _dataIdentifier
   */
  std::string _dataID;

  /**
   * local Copy of Content for the single Points
   */
  std::list<float> _vertexDataWriterContentList;
#ifdef Parallel
  /**
   * Output stream
   */
  std::ostringstream _outPVTU;

#endif

  int _recordsPerVertex;

  double _minValue;
  double _maxValue;

  VertexDataWriter(const std::string& dataIdentifier, BinaryVTUTextFileWriter& writer, int recordsPerVertex);

#ifdef Asserts
  /**
   * There is no reason to store a data identifier. However, we augment
   * the assertions with the identifier to make it a little bit easier to
   * find data inconsistencies.
   */
  std::string  _dataIdentifier;
#endif
public:
  virtual ~VertexDataWriter();

  virtual void close();

  virtual void plotVertex( int index, double value );
  virtual void plotVertex( int index, const tarch::la::Vector<2,double>& value );
  virtual void plotVertex( int index, const tarch::la::Vector<3,double>& value );

  virtual double getMinValue() const;
  virtual double getMaxValue() const;

  void assignRemainingVerticesDefaultValues() {}
};
};

#endif
