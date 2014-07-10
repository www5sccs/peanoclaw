#ifndef __VCC_DEM_H__
#define __VCC_DEM_H__

// DEM v 1.0.2

// TODO
// * out-of-core, tile-major with 1 pixel boundary

#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<memory>
#include<cstring>

#include <cmath>

class DEM {
public:
							DEM(void);
							DEM(const DEM& other);
							~DEM(void);
	inline	bool			resize(int width, int height, int boundary=0);
	inline	void			clear(void);
	inline	bool			fill(const float& val);
	inline	bool			zero(void);

	/// boundary access
	inline	const int&		boundary_size(void) const;
	inline	bool			fillBoundary(const float& val);
	inline	bool			zeroBoundary(void);
	
	/// data access at grid points (size_t) and interpolated (const double&), all in pixel-space
	inline	float&			operator()(int i, int j);
	inline	const float&	operator()(int i, int j) const;
	inline	float			operator()(const double x, const double y) const;
	
	/// direct access of grid
	inline	float&			operator[](size_t addr);
	inline	const float&	operator[](size_t addr) const;
	
	/// dimensions and geo-references
	inline	size_t			nPixels(void) const;
	inline	const int&		dimension(int n) const;
	inline	double&			lower_left(int n);
	inline	const double&	lower_left(int n) const;
	inline	double&			upper_right(int n);
	inline	const double&	upper_right(int n) const;
	inline	double			scale(int n) const;

	// compute world-space gradients at pixel-space coordinates i,j or x,y
	inline	void			gradient(float& gx, float& gy, int i, int j) const;
	inline	void			gradient(float& gx, float& gy, const double& x, const double& y) const;

	/// IO
			bool			save(const std::string& name) const;
			bool			save(FILE* stream) const;
			bool			load(const std::string& name);
			bool			load(FILE* stream);
			bool			exportOBJ(const std::string& name) const;
			bool			exportOBJ(FILE* stream) const;
			bool			exportCSV(const std::string& name) const;
			bool			exportCSV(FILE* stream) const;
			bool			exportESRI(const std::string& name) const;
			bool			exportESRI(FILE* stream) const;

	/// further operators and methods
	inline	bool			isValid(void) const;
	inline	const DEM&		operator=(const DEM& other);
	inline	bool			minmax(void);
	inline	bool			cropROI(DEM& other, int x, int y, int width, int height, int boundary=0) const;

	/// world-space to pixel-space conversion
	inline	void			worldspace_to_pixelspace(double& psx, double& psy, const double& wsx, const double& wsy) const;
	inline	void			pixelspace_to_worldspace(double& wsx, double& wsy, const double& psx, const double& psy) const;
	inline	void			pixelspace_to_worldspace(double& wsx, double& wsy, int i, int j) const;
	inline	void			address_to_worldspace(double& wsx, double& wsy, size_t addr) const;
	static	const float		INVALID;
protected:
			float*			m_data;
			float*			m_boundary_data;
			float*			m_bound_top;
			float*			m_bound_left;
			float*			m_bound_right;
			float*			m_bound_bottom;
			int				m_boundary_size;
			int				m_dimension[2];
			double			m_LowerLeft[3];
			double			m_UpperRight[3];
	inline	size_t			nBoundaryPixels(void) const;
};

inline bool DEM::resize(int width, int height, int boundary) {
	size_t nSize = size_t(width)*size_t(height);
	size_t nBoundarySize = nBoundaryPixels();
	if (m_dimension[0]*m_dimension[1]!=nSize) {
		if (m_data!=NULL) delete[] m_data;
		if (nSize==0) {
			m_data = NULL;
			m_dimension[0] = 0;
			m_dimension[1] = 0;
			return true;
		}
		m_data = new float[nSize];
		if (m_data==NULL) return false;
	}	
	m_dimension[0] = width;
	m_dimension[1] = height;
	m_boundary_size= boundary;
	if (nBoundaryPixels()!=nBoundarySize) {
		if (m_boundary_data) delete[] m_boundary_data;
		if (nBoundaryPixels()!=0) m_boundary_data = new float[nBoundaryPixels()];		
	}
	if (m_boundary_data) {
		m_bound_bottom	=  m_boundary_data;
		m_bound_left	= &m_bound_bottom  [m_boundary_size*(m_dimension[0]+2*m_boundary_size)];		
		m_bound_right	= &m_bound_left [m_boundary_size*m_dimension[1]];
		m_bound_top		= &m_bound_right[m_boundary_size*m_dimension[1]];
	}
	return true;
}

inline void DEM::clear(void) {
	if (m_data!=NULL)		delete[] m_data;				m_data = NULL;
	if (m_boundary_data)	delete[] m_boundary_data;		m_boundary_data = NULL;
	m_dimension[0] = 0;
	m_dimension[1] = 0;
}

inline bool DEM::fill(const float& val) {
	if (!isValid()) return false;
	for (size_t n=0; n<nPixels(); n++) m_data[n] = val;
	return true;
}

inline bool DEM::zero(void) {
	return fill(0.0f);
}

inline bool DEM::fillBoundary(const float& val) {
	if (!isValid()) return false;
	for (size_t n=0; n<nBoundaryPixels(); n++) m_boundary_data[n] = val;
	return true;
}

inline bool DEM::zeroBoundary(void) {
	return fill(0.0f);
}
	

inline float& DEM::operator()(int i, int j) {
	// boundary treatment
	if (m_boundary_size>0) {
		if (j<0) {
			return m_bound_bottom[i+m_boundary_size + (j+m_boundary_size)*(m_dimension[0]+2*m_boundary_size) ];			
		}
		else if (j>=m_dimension[1]) {
			return m_bound_top[i+m_boundary_size +(j-m_dimension[1])*(m_dimension[0]+2*m_boundary_size) ];			
		}
		else if (i<0) {			
			return m_bound_left[i+m_boundary_size + j*m_boundary_size];
		}
		else if (i>=m_dimension[0]) {
			return m_bound_right[i-m_dimension[0] + j*m_boundary_size];			
		}
	}
	return m_data[size_t(i)+size_t(j)*size_t(m_dimension[0])];
}

inline const float&	DEM::operator()(int i, int j) const {
	// boundary treatment
	if (m_boundary_size>0) {
		if (j<0) {
			return m_bound_bottom[i+m_boundary_size + (j+m_boundary_size)*(m_dimension[0]+2*m_boundary_size) ];			
		}
		else if (j>=m_dimension[1]) {
			return m_bound_top[i+m_boundary_size +(j-m_dimension[1])*(m_dimension[0]+2*m_boundary_size) ];			
		}
		else if (i<0) {
			return m_bound_left[i+m_boundary_size + j*m_boundary_size];
		}
		else if (i>=m_dimension[0]) {
			return m_bound_right[i-m_dimension[0] + j*m_boundary_size];			
		}
	}
	return m_data[size_t(i)+size_t(j)*size_t(m_dimension[0])];
}

inline float DEM::operator()(const double x, const double y) const {
	if (x<0.0 || x>double(m_dimension[0]-1)) return INVALID;
	if (y<0.0 || y>double(m_dimension[1]-1)) return INVALID;
	size_t i = size_t(std::floor(x));
	size_t j = size_t(std::floor(y));
	double wx = x-std::floor(x);
	double wy = y-std::floor(y);
	size_t i1 = std::min<size_t>(i+1,m_dimension[0]-1);
	size_t j1 = std::min<size_t>(j+1,m_dimension[1]-1);

	double t0 = m_data[i+j *m_dimension[0]] + wx*(m_data[i1+j *m_dimension[0]]-m_data[i+j *m_dimension[0]]);
	double t1 = m_data[i+j1*m_dimension[0]] + wx*(m_data[i1+j1*m_dimension[0]]-m_data[i+j1*m_dimension[0]]);
	return float(t0 + wy*(t1-t0));
}
	
inline float& DEM::operator[](size_t addr) {
	return m_data[addr];
}

inline const float&	DEM::operator[](size_t addr) const {
	return m_data[addr];
}

inline size_t DEM::nPixels(void) const {
	return m_dimension[0]*m_dimension[1];
}

inline const int& DEM::dimension(int n) const {
	return m_dimension[n];
}

inline double& DEM::lower_left(int n) {
	return m_LowerLeft[n];
}

inline const double& DEM::lower_left(int n) const {
	return m_LowerLeft[n];
}

inline double& DEM::upper_right(int n) {
	return m_UpperRight[n];
}

inline const double& DEM::upper_right(int n) const {
	return m_UpperRight[n];
}

inline double DEM::scale(int n) const {
	return (m_UpperRight[n]-m_LowerLeft[n])/double(m_dimension[n]-1);
}

inline bool DEM::isValid(void) const {
	return m_data!=NULL;
}

inline const DEM& DEM::operator=(const DEM& other) {
	if (!resize(other.dimension(0),other.dimension(1))) {
		clear();
		return *this;
	}
	memcpy(m_data,other.m_data,sizeof(float)*other.nPixels());
	for (int i=0; i<3; i++) {
		m_LowerLeft[i] = other.m_LowerLeft[i];
		m_UpperRight[i]= other.m_UpperRight[i];
	}
	return *this;
}

inline bool DEM::minmax(void) {
	if (!isValid()) return false;
	m_LowerLeft[2] = m_data[0];
	m_UpperRight[2]= m_data[0];
	for (size_t n=1; n<nPixels(); n++) {
		m_LowerLeft[2] = std::min<double>(double(m_data[n]),m_LowerLeft[2]);
		m_UpperRight[2]= std::max<double>(double(m_data[n]),m_UpperRight[2]);
	}
	return true;
}

inline void DEM::worldspace_to_pixelspace(double& psx, double& psy, const double& wsx, const double& wsy) const {
	psx = (wsx-m_LowerLeft[0])/(m_UpperRight[0]-m_LowerLeft[0])*double(m_dimension[0]-1);
	psy = (wsy-m_LowerLeft[1])/(m_UpperRight[1]-m_LowerLeft[1])*double(m_dimension[1]-1);
}

inline void DEM::pixelspace_to_worldspace(double& wsx, double& wsy, const double& psx, const double& psy) const {
	wsx = psx/double(m_dimension[0]-1)*(m_UpperRight[0]-m_LowerLeft[0])+m_LowerLeft[0];
	wsy = psy/double(m_dimension[1]-1)*(m_UpperRight[1]-m_LowerLeft[1])+m_LowerLeft[1];
}

inline void DEM::pixelspace_to_worldspace(double& wsx, double& wsy, int i, int j) const {
	return pixelspace_to_worldspace(wsx,wsy,double(i),double(j));
}

inline void DEM::address_to_worldspace(double& wsx, double& wsy, size_t addr) const {
	return pixelspace_to_worldspace(wsx,wsy,int(addr%size_t(m_dimension[0])),int(addr/size_t(m_dimension[0])));
}

inline size_t DEM::nBoundaryPixels(void) const {
	return size_t(m_dimension[0]+2*m_boundary_size+m_dimension[1])*2*size_t(m_boundary_size);
}

inline const int& DEM::boundary_size(void) const {
	return m_boundary_size;
}

inline void DEM::gradient(float& gx, float& gy, int i, int j) const {
	int il = std::max<int>(i-1,0);
	int ir = std::min<int>(i+1,m_dimension[0]-1);
	int jl = std::max<int>(j-1,0);
	int jr = std::min<int>(j+1,m_dimension[1]-1);
	gx = float((m_data[ir+j*m_dimension[0]]-m_data[il+j*m_dimension[0]])/double(ir-il)/scale(0));
	gy = float((m_data[i+jr*m_dimension[0]]-m_data[i+jl*m_dimension[0]])/double(jr-jl)/scale(1));
}

inline void DEM::gradient(float& gx, float& gy, const double& x, const double& y) const {
	float G00[2], G01[2], G10[2], G11[2];
	int ix = std::max<int>(0,int(floor(x)));
	int ix1= std::min<int>(m_dimension[0]-1,ix+1);
	float wx = float(x-floor(x));
	int iy = std::max<int>(0,int(floor(y)));
	int iy1= std::min<int>(m_dimension[1]-1,iy+1);
	float wy = float(y-floor(y));
	gradient(G00[0],G00[1],ix ,iy );
	gradient(G01[0],G01[1],ix1,iy );
	gradient(G10[0],G10[1],ix1,iy1);
	gradient(G11[0],G11[1],ix1,iy1);

	G00[0] = G00[0] + wx*(G01[0]-G00[0]);
	G00[1] = G00[1] + wx*(G01[1]-G00[1]);
	G10[0] = G10[0] + wx*(G11[0]-G10[0]);
	G10[1] = G10[1] + wx*(G11[1]-G10[1]);
	gx = G00[0] + wy*(G10[0]-G00[0]);
	gy = G00[1] + wy*(G10[1]-G00[1]);
}

inline bool DEM::cropROI(DEM& output, int x, int y, int width, int height, int boundary) const {
	if (x+width>m_dimension[0] || y+height>m_dimension[1]) return false;
	if (x<0 || y<0 || width<0 || height<0) return false;
	if (!output.resize(width,height,boundary)) return false;
	for (int j=0; j<width; j++) {
		for (int i=0; i<width; i++) {
			output(i,j) = m_data[(x+i)+(y+j)*m_dimension[0]];
		}
	}
	pixelspace_to_worldspace(output.lower_left(0),output.lower_left(1),x,y);
	pixelspace_to_worldspace(output.upper_right(0),output.upper_right(1),x+width-1,y+height-1);
	output.minmax();
	return true;
}

#endif
