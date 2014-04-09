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

typedef double  (*RefinementCriterionCallback)(PyObject* q,
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

typedef void (*InterPatchCommunicationCallback)(
                        PyObject* sourceQ,
                        PyObject* sourceQBC,
                        PyObject* sourceAux,
                        int sourceSubdivisionFactorX0,
                        int sourceSubdivisionFactorX1,
                        int sourceSubdivisionFactorX2,
                        double sourceSizeX,
                        double sourceSizeY,
                        double sourceSizeZ,
                        double sourcePositionX,
                        double sourcePositionY,
                        double sourcePositionZ,
                        double sourceCurrentTime,
                        double sourceTimestepSize,
                        PyObject* destinationQ,
                        PyObject* destinationQBC,
                        PyObject* destinationAux,
                        int destinationSubdivisionFactorX0,
                        int destinationSubdivisionFactorX1,
                        int destinationSubdivisionFactorX2,
                        double destinationSizeX,
                        double destinationSizeY,
                        double destinationSizeZ,
                        double destinationPositionX,
                        double destinationPositionY,
                        double destinationPositionZ,
                        double destinationCurrentTime,
                        double destinationTimestepSize,
                        int unknownsPerSubcell,
                        int auxFieldsPerSubcell);

#endif // PYCLAWCALLBACKS_H
