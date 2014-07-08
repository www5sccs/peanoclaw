#include"dem.h"

const float DEM::INVALID = -1e38f;

DEM::DEM(void) : m_data(NULL), m_boundary_data(NULL), m_bound_bottom(NULL), m_bound_top(NULL), m_bound_left(NULL), m_bound_right(NULL) {
	m_dimension[0]	= m_dimension[1]	= 0;
	m_LowerLeft[0]	= m_LowerLeft[1]	= m_LowerLeft[2] = INVALID;
	m_UpperRight[0]	= m_UpperRight[1]	= m_UpperRight[2]= INVALID;
	m_boundary_size = 0;
}

DEM::~DEM(void) {
	clear();
}

DEM::DEM(const DEM& other) : m_data(NULL), m_boundary_data(NULL), m_bound_bottom(NULL), m_bound_top(NULL), m_bound_left(NULL), m_bound_right(NULL) {
	m_dimension[0] = m_dimension[1] = 0;
	*this = other;
}

bool DEM::save(const std::string& name) const {
	FILE* fptr = NULL;
#ifdef _MSC_VER
	if (fopen_s(&fptr,name.c_str(),"wb")) return false;
#else 
	fptr = fopen(name.c_str(),"wb");
	if (fptr==NULL) return false;
#endif
	bool bRes = save(fptr);
	fclose(fptr);
	return bRes;
}

bool DEM::save(FILE* stream) const {
	if (stream==NULL) return false;
	int dims[3];
	dims[0] = int(m_dimension[0]);
	dims[1] = int(m_dimension[1]);
	dims[2] = int(m_boundary_size);
	if (fwrite(dims,sizeof(int)*3,1,stream)!=1)					return false;
	if (fwrite(m_LowerLeft,sizeof(double)*3,1,stream)!=1)		return false;
	if (fwrite(m_UpperRight,sizeof(double)*3,1,stream)!=1)		return false;
	
	const size_t chunk = (8<<20)/sizeof(float);	// 8MB chunks
	size_t size = nPixels();
	float* ptr = m_data;
	while (size>0) {
		size_t toWrite = std::min<size_t>(chunk,size);
		if (fwrite(ptr,sizeof(float)*toWrite,1,stream)!=1)		return false;
		size-=toWrite;
		ptr+=toWrite;
	}

	if (dims[2]>0) {
		float *ptr = m_boundary_data;
		size = nBoundaryPixels();
		while (size>0) {
			size_t toWrite = std::min<size_t>(chunk,size);
			if (fwrite(ptr,sizeof(float)*toWrite,1,stream)!=1)	return false;
			size-=toWrite;
			ptr+=toWrite;
		}
	}
	return true;
}

bool DEM::load(const std::string& name) {
	FILE* fptr = NULL;
#ifdef _MSC_VER
	if (fopen_s(&fptr,name.c_str(),"rb")) return false;
#else
	fptr = fopen(name.c_str(),"rb");
	if (fptr==NULL) return false;
#endif
	bool bRes = load(fptr);
	fclose(fptr);
	return bRes;
}

bool DEM::load(FILE* stream) {
	if (stream==NULL) return false;
	clear();
	int dims[3];
	if (fread(dims,sizeof(int)*3,1,stream)!=1)					return false;
	m_dimension[0] = dims[0];
	m_dimension[1] = dims[1];
	m_boundary_size= dims[2];
	if (fread(m_LowerLeft,sizeof(double)*3,1,stream)!=1)		return false;
	if (fread(m_UpperRight,sizeof(double)*3,1,stream)!=1)		return false;
	m_data = new float[nPixels()];
	if (!m_data) return false;
	const size_t chunk = (8<<20)/sizeof(float); // 8MB chunks
	size_t size = nPixels();
	float* ptr = m_data;
	while (size>0) {
		size_t toRead = std::min<size_t>(chunk,size);
		if (fread(ptr,sizeof(float)*toRead,1,stream)!=1)		return false;
		size-=toRead;
		ptr+=toRead;
	}
	if (m_boundary_size>0) {
		size_t size = nBoundaryPixels();
		m_boundary_data = new float[size];
		if (!m_boundary_data) return false;
		float* ptr = m_boundary_data;
		while (size>0) {
			size_t toRead = std::min<size_t>(chunk,size);
			if (fread(ptr,sizeof(float)*toRead,1,stream)!=1)	return false;
			size-=toRead;
			ptr+=toRead;
		}
		m_bound_bottom	=  m_boundary_data;
		m_bound_left	= &m_bound_bottom[m_boundary_size*(m_dimension[0]+2*m_boundary_size)];
		m_bound_right	= &m_bound_left[m_boundary_size*m_dimension[1]];
		m_bound_top		= &m_bound_right[m_boundary_size*m_dimension[1]];
	}
	return true;
}

bool DEM::exportOBJ(const std::string& name) const {
	FILE* fptr = NULL;
#ifdef _MSC_VER
	if (fopen_s(&fptr,name.c_str(),"wt")) return false;
#else
	fptr = fopen(name.c_str(),"wt");
	if (fptr==NULL) return false;
#endif
	bool bRes = exportOBJ(fptr);
	fclose(fptr);
	return bRes;
}

bool DEM::exportOBJ(FILE* fptr) const {
	if (fptr==NULL) return false;
	if (!isValid()) return false;
	for (int j=0; j<m_dimension[1]; j++) {
		for (int i=0; i<m_dimension[0]; i++) {
			double x,y;
			pixelspace_to_worldspace(x,y,i,j);			
			fprintf(fptr,"v %g %g %g\n",x,y,m_data[i+j*m_dimension[0]]);
		}
	}
	for (int j=0; j<m_dimension[1]-1; j++) {
		for (int i=0; i<m_dimension[0]-1; i++) {
			int idA = i+  j  *m_dimension[0],	idB = idA+1;
			int idC = i+(j+1)*m_dimension[0],	idD = idC+1;
			float AD = std::abs(m_data[idA]-m_data[idD]);
			float BC = std::abs(m_data[idB]-m_data[idC]);
			if (AD<BC) {
				// triangles abd, adc
				fprintf(fptr,"f %i %i %i\n",idA+1,idB+1,idD+1);
				fprintf(fptr,"f %i %i %i\n",idA+1,idD+1,idC+1);
			} else {
				// triangles abc, bdc
				fprintf(fptr,"f %i %i %i\n",idA+1,idB+1,idC+1);
				fprintf(fptr,"f %i %i %i\n",idB+1,idD+1,idC+1);
			}
		}
	}
	return false;
}

bool DEM::exportCSV(const std::string& name) const {
	FILE* fptr = NULL;
#ifdef _MSC_VER
	if (fopen_s(&fptr,name.c_str(),"wt")) return false;
#else
	fptr = fopen(name.c_str(),"wt");
	if (fptr==NULL) return false;
#endif
	bool bRes = exportCSV(fptr);
	fclose(fptr);
	return bRes;
}

bool DEM::exportCSV(FILE* fptr) const {
	if (fptr==NULL) return false;
	if (!isValid()) return false;
	fprintf(fptr,"DEM east-major\n");
	fprintf(fptr,"width,%i,height,%i\n",m_dimension[0],m_dimension[1]);	
	fprintf(fptr,"minEast,%g,maxEast,%g\n",m_LowerLeft[0],m_LowerLeft[1]);
	fprintf(fptr,"minNorth,%g,maxNorth,%g\n",m_UpperRight[0],m_UpperRight[1]);
	fprintf(fptr,"scaleEast,%g,scaleNorth,%g\n\n\n\n\n",scale(0),scale(1));
	for (int j=0; j<m_dimension[1]; j++) {
		for (int i=0; i<m_dimension[0]; i++) {
			fprintf(fptr,"%g%s",m_data[i+j*m_dimension[0]], i+1==m_dimension[0] ? "\n" : ",");
		}
	}
	return true;
}

bool DEM::exportESRI(const std::string& name) const {
	FILE* fptr = NULL;
#ifdef _MSC_VER
	if (fopen_s(&fptr,name.c_str(),"wt")) return false;
#else
	fptr = fopen(name.c_str(),"wt");
	if (fptr==NULL) return false;
#endif
	bool bRes = exportESRI(fptr);
	fclose(fptr);
	return bRes;
}

bool DEM::exportESRI(FILE* stream) const {
	if (stream==NULL) return false;
	printf("SCALE: %g,%g\n",scale(0),scale(1));
	// TODO: resample data for uniform scale
	fprintf(stream,"ncols     %i\n",m_dimension[0]);
	fprintf(stream,"nrows     %i\n",m_dimension[1]);
	fprintf(stream,"xllcorner %g\n",lower_left(0));
	fprintf(stream,"yllcorner %g\n",lower_left(1));
	fprintf(stream,"cellsize  %g\n",scale(0));
	fprintf(stream,"NODATA_value -9999\n");
	for (int j=m_dimension[1]-1; j>=0; j--) {
		for (int i=0; i<m_dimension[0]; i++) {
			fprintf(stream,"%f%s",m_data[i+j*m_dimension[0]],i+1==m_dimension[0] ? "\n" : " ");
		}
	}
	return true;
}
