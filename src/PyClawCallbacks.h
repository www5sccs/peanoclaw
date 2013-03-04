#ifndef PYCLAWCALLBACKS_H
#define PYCLAWCALLBACKS_H

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

typedef double  (*InitializationCallback)(PyObject* q,
                                         PyObject* qbc,
                                         PyObject* aux,
                                         int subdivisionFactorX0,
                                         int subdivisionFactorX1,
                                         int subdivisionFactorX2,
                                         int unknownsPerSubcell,
                                         int auxFieldsPerSubcell,
                                         double sizeX,
                                         double sizeY,
                                         double sizeZ,
                                         double positionX,
                                         double positionY,
                                         double positionZ);

typedef void (*BoundaryConditionCallback)(PyObject* q,
                                          PyObject* qbc,
                                          int dimension,
                                          int setUpper);

typedef double (*SolverCallback)(double* dtAndCfl,
                        PyObject* q,
                        PyObject* qbc,
                        PyObject* aux,
                        int subdivisionFactorX0,
                        int subdivisionFactorX1,
                        int subdivisionFactorX2,
                        int unknownsPerSubcell,
                        int auxFieldsPerSubcell,
                        double sizeX,
                        double sizeY,
                        double sizeZ,
                        double positionX,
                        double positionY,
                        double positionZ,
                        double currentTime,
                        double maximumTimestepSize,
                        double estimatedNextTimestepSize,
                        bool useDimensionalSplitting);


//------------------------------------------


typedef void (*AddPatchToSolutionCallback)(PyObject* q,
                                    PyObject* qbc,
                                    int ghostlayerWidth,
                                    double sizeX,
                                    double sizeY,
                                    double sizeZ,
                                    double positionX,
                                    double positionY,
                                    double positionZ,
                                    double currentTime);

#endif // PYCLAWCALLBACKS_H
