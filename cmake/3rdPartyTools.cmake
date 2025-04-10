#  This file configures which 3rd party tools are built in and which are used from the system.
#  NOTE: must be included after WhichTools.cmake

message(STATUS "Checking whether to use built-in libraries...")

if(PMEMD_ONLY)

set(3RDPARTY_TOOLS
blas
lapack
netcdf
netcdf-fortran
zlib
libbz2
boost
kmmd
libm)

set(3RDPARTY_TOOL_USES
"for fundamental linear algebra calculations"
"for fundamental linear algebra calculations"
"for creating trajectory data files"
"for creating trajectory data files from Fortran"
"for various compression and decompression tasks"
"for various compression and decompression tasks"
"for supporting the gamma distribution"
"Machine-learning molecular dynamics"
"for fundamental math routines if they are not contained in the C library")

else()

# List of valid 3rd party tools.  All these names should be all lowercase.
set(3RDPARTY_TOOLS
blas
lapack
arpack
ucpp
readline
c9x-complex
netcdf
netcdf-fortran
pnetcdf
protobuf
fftw
xblas
lio
apbs
pupil
zlib
libbz2
plumed
libm
mkl
kmmd
mpi4py
perlmol
boost
nccl
mbx
tng_io
torchani
nlopt
libtorch
xtb
dftbplus
deepmd-kit)

set(3RDPARTY_TOOL_USES
"for fundamental linear algebra calculations"
"for fundamental linear algebra calculations"
"for fundamental linear algebra calculations"
"used as a preprocessor for the NAB compiler"
"enables an interactive terminal in cpptraj"
"used as a support library on systems that do not have C99 complex.h support"
"for creating trajectory data files"
"for creating trajectory data files from Fortran"
"used by cpptraj for parallel trajectory output"
"protocol buffers library, used for communication with external software in QM/MM"
"used to do Fourier transforms very quickly"
"used for high-precision linear algebra calculations"
"used by Sander to run certain QM routines on the GPU"
"used by Sander as an alternate Poisson-Boltzmann equation solver"
"used by Sander as an alternate user interface"
"for various compression and decompression tasks"
"for bzip2 compression in cpptraj"
"used as an alternate MD backend for Sander"
"for fundamental math routines if they are not contained in the C library"
"alternate implementation of lapack and blas that is tuned for speed"
"Machine-learning molecular dynamics"
"MPI support library for MMPBSA.py"
"chemistry library used by FEW"
"C++ support library"
"NVIDIA parallel GPU communication library"
"computes energies and forces for pmemd with the MB-pol model"
"enables GROMACS tng trajectory input in cpptraj"
"enables computation of energies and forces with Torchani"
"used to perform nonlinear optimizations"
"enables libtorch C++ library for tensor computation and dynamic neural networks"
"enables the QM/MM interface with the external xtb software"
"enables the QM/MM interface with the external dftbplus software"
"enables DPRc machine learning potential with the external deepmd-kit software")

endif()

# Logic to disable tools
set(3RDPARTY_SUBDIRS "")

#sets a tool to external, internal, or disabled
#STATUS=EXTERNAL, INTERNAL, or DISABLED
macro(set_3rdparty TOOL STATUS)
	set(${TOOL}_INTERNAL FALSE)
	set(${TOOL}_EXTERNAL FALSE)
	set(${TOOL}_DISABLED FALSE)
	set(${TOOL}_ENABLED TRUE)


	if(${STATUS} STREQUAL EXTERNAL)
		set(${TOOL}_EXTERNAL TRUE)
	elseif(${STATUS} STREQUAL INTERNAL)

		#the only way to get this message would be to use FORCE_INTERNAL_LIBS incorrectly, unless someone messed up somewhere
		if("${BUNDLED_3RDPARTY_TOOLS}" MATCHES ${TOOL})
				set(${TOOL}_INTERNAL TRUE)
		else()
			if(INSIDE_AMBER)
				# getting here means there's been a programming error
				message(FATAL_ERROR "3rd party program ${TOOL} is not bundled and cannot be built inside Amber.")
			elseif("${REQUIRED_3RDPARTY_TOOLS}" MATCHES ${TOOL})
				# kind of a kludge - even when we're in a submodule, things will still get set to internal, so we just treat internal equal to disabled
				message(FATAL_ERROR "3rd party program ${TOOL} is required, but was not found.")
			else()
				# we're in a submodule, and it's not required, so it's OK that the tool is not bundled
				set(${TOOL}_DISABLED TRUE)
				set(${TOOL}_ENABLED FALSE)

			endif()
		endif()

	else()
		list_contains(TOOL_REQUIRED ${TOOL} ${REQUIRED_3RDPARTY_TOOLS})

		if(TOOL_REQUIRED)
			message(FATAL_ERROR "3rd party program ${TOOL} is required to build ${PROJECT_NAME}, but it is disabled (likely because it was not found).")
		endif()

		set(${TOOL}_DISABLED TRUE)
		set(${TOOL}_ENABLED FALSE)

	endif()
endmacro(set_3rdparty)

#------------------------------------------------------------------------------
#  OS threading library (not really a 3rd party tool)
#------------------------------------------------------------------------------
set(CMAKE_THREAD_PREFER_PTHREAD TRUE) #Yeah, we're biased.
find_package(Threads)

# first, figure out which tools we need
# -------------------------------------------------------------------------------------------------------------------------------

# if NEEDED_3RDPARTY_TOOLS is not passed in, assume that all of them are needed
if(NOT DEFINED NEEDED_3RDPARTY_TOOLS)
	set(NEEDED_3RDPARTY_TOOLS "${3RDPARTY_TOOLS}")
endif()

if(NOT DEFINED BUNDLED_3RDPARTY_TOOLS)
	set(BUNDLED_3RDPARTY_TOOLS "")
endif()

# suspicious 3rd party tools: tools where there are often problems with the system install
if(NOT DEFINED SUSPICIOUS_3RDPARTY_TOOLS)
	set(SUSPICIOUS_3RDPARTY_TOOLS "mkl")
endif()

option(TRUST_SYSTEM_LIBS "If true, Amber will use all found system libraries, even if they are considered problematic by the Amber developers" FALSE)

foreach(TOOL ${3RDPARTY_TOOLS})
	list(FIND NEEDED_3RDPARTY_TOOLS ${TOOL} TOOL_INDEX)

	test(NEED_${TOOL} NOT "${TOOL_INDEX}" EQUAL -1)
endforeach()

if(("${NEEDED_3RDPARTY_TOOLS}" MATCHES "mkl" OR "${NEEDED_3RDPARTY_TOOLS}" MATCHES "blas" OR "${NEEDED_3RDPARTY_TOOLS}" MATCHES "lapack")
	AND NOT ("${NEEDED_3RDPARTY_TOOLS}" MATCHES "mkl" AND "${NEEDED_3RDPARTY_TOOLS}" MATCHES "blas" AND "${NEEDED_3RDPARTY_TOOLS}" MATCHES "lapack"))
	message(FATAL_ERROR "If any of mkl, blas, and lapack are put into NEEDED_3RDPARTY_TOOLS, them you must put all of them in since mkl replaces blas and lapack")
endif()

# 1st pass checking
# -------------------------------------------------------------------------------------------------------------------------------

# Force all try_link_library() calls to build position independent executables if Amber is built as shared.
# This ensures that all dependent libraries are built as position independent
set(CMAKE_POSITION_INDEPENDENT_CODE ${SHARED})


#------------------------------------------------------------------------------
# check if we need to use c9xcomplex
#------------------------------------------------------------------------------

if(NEED_c9x-complex)
	check_include_file(complex.h LIBC_HAS_COMPLEX)
	if(LIBC_HAS_COMPLEX)
		set_3rdparty(c9x-complex DISABLED)
	else()
		set_3rdparty(c9x-complex INTERNAL)
	endif()
endif()

#------------------------------------------------------------------------------
# check for ucpp
#------------------------------------------------------------------------------
if(NEED_ucpp)
	find_program(UCPP_LOCATION ucpp)

	if(UCPP_LOCATION)
		set_3rdparty(ucpp EXTERNAL)
	else()
		set_3rdparty(ucpp INTERNAL)
	endif()
endif()

#------------------------------------------------------------------------------
#  Readline
#------------------------------------------------------------------------------

if(NEED_readline)
	find_package(Readline)

	if(READLINE_FOUND)
		set_3rdparty(readline EXTERNAL)
	elseif(NOT TARGET_WINDOWS)
		set_3rdparty(readline INTERNAL)
	else()
		set_3rdparty(readline DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  MKL (near the top because it contains lapack, and blas)
#------------------------------------------------------------------------------

if(NEED_mkl)

	set(MKL_NEEDINCLUDES FALSE)
	set(MKL_NEEDEXTRA FALSE)

	# Static MKL is not the default at this time.
	# <long_explanation>
	# MKL has a fftw3 compatibility interface.  Weirdly enough, this interface is spread out between several different libraries: the main interface library, the
	# cdft library, and the actual fftw3 interface library (which is distributed as source code, not a binary).
	# So, even though we don't use the fftw3 interface, there are symbols in the main MKL libraries which conflict with the symbols from fftw3.
	# Oddly, on many platforms, the linker handles this fine.  However, in at least one case (the SDSC supercomputer Comet, running a derivative of CentOS),
	# ld balks at this multiple definition, and refuses to link programs which use MKL and fftw, but ONLY when BOTH of them are built as static libraries.
	# Why this is, I'm not sure.  I do know that it's better to build fftw3 as static and use mkl as shared (because mkl is a system library)
	# then the other way around, so that's what I do
	# </long_explanation>
	option(MKL_STATIC "Whether to prefer MKL's static libraries" FALSE)

	option(MKL_MULTI_THREADED "Whether to link MKL in OpenMP mode to parallelize singlethreaded calls to MKL functions" TRUE)

	set(MKL_STATIC FALSE)
	find_package(MKL)

	if(MKL_FOUND)
		set_3rdparty(mkl EXTERNAL)
	else()
		set_3rdparty(mkl DISABLED)
	endif()
endif()

if(NEED_kmmd)
    find_package(KMMD)
	if(KMMD_FOUND)
	   set_3rdparty(kmmd EXTERNAL)
	else()
	   set_3rdparty(kmmd INTERNAL)
	   set(KMMD_LIB kmmd)
	endif()

endif()



#------------------------------------------------------------------------------
#  FFTW
#------------------------------------------------------------------------------
if(NEED_fftw)

	if(DEFINED USE_FFT AND NOT USE_FFT)
		set_3rdparty(fftw DISABLED)
	else()
		if(MPI)
			find_package(FFTW COMPONENTS MPI Fortran)
		else()
			find_package(FFTW COMPONENTS Fortran)
		endif()

		if(FFTW_FOUND)
			set_3rdparty(fftw EXTERNAL)
		else()
			set_3rdparty(fftw INTERNAL)
		endif()
	endif()
endif()

#------------------------------------------------------------------------------
#  NetCDF
#------------------------------------------------------------------------------

if(NEED_netcdf OR NEED_netcdf-fortran)
	find_package(NetCDF OPTIONAL_COMPONENTS F77 F90)
endif()

if(NEED_netcdf)
	if(NetCDF_FOUND)
		set_3rdparty(netcdf EXTERNAL)
	else()
		set_3rdparty(netcdf INTERNAL)
	endif()
endif()

if(NEED_netcdf-fortran)
	if(NetCDF_F90_FOUND)
		set_3rdparty(netcdf-fortran EXTERNAL)
	else()
		set_3rdparty(netcdf-fortran INTERNAL)
	endif()
endif()

#------------------------------------------------------------------------------
#  Protobuf
#------------------------------------------------------------------------------

if(NEED_protobuf)
	find_package(Protobuf)

	if(Protobuf_FOUND)
		if(BUILD_PROTOBUF)
		    set_3rdparty(protobuf EXTERNAL)
	    else()
    	    set_3rdparty(protobuf DISABLED)
	    endif()
	else()
		if(BUILD_PROTOBUF)
		    set_3rdparty(protobuf INTERNAL)
	    else()
    	    set_3rdparty(protobuf DISABLED)
	    endif()
	endif()
endif()

#------------------------------------------------------------------------------
#  XBlas
#------------------------------------------------------------------------------

if(NEED_xblas)

	find_package(XBLAS)

	if(XBLAS_FOUND)
		set_3rdparty(xblas EXTERNAL)
	else()
		if(EXISTS "${M4}")
			set_3rdparty(xblas INTERNAL)
		else()
			message(STATUS "Internal xblas cannot build since m4 was not found.")
			set_3rdparty(xblas DISABLED)
		endif()
	endif()
endif()

#------------------------------------------------------------------------------
#  Netlib libraries
#------------------------------------------------------------------------------

if(NEED_blas) # because of the earlier check, we can be sure that NEED_blas == NEED_lapack

	# this calls FindBLAS
	find_package(LAPACKFixed)

	if(BLAS_FOUND)
		set_3rdparty(blas EXTERNAL)
	else()
		set_3rdparty(blas INTERNAL)
	endif()

	if(LAPACK_FOUND)
		set_3rdparty(lapack EXTERNAL)
	else()
		set_3rdparty(lapack INTERNAL)
	endif()
endif()

if(NEED_arpack)
	#  ARPACK
	find_package(ARPACK)

	if(ARPACK_FOUND)
		set_3rdparty(arpack EXTERNAL)
	else()
		set_3rdparty(arpack INTERNAL)
	endif()
endif()

# --------------------------------------------------------------------
#  Parallel NetCDF
# --------------------------------------------------------------------

if(NEED_pnetcdf)
	find_package(PnetCDF)

	if(PnetCDF_FOUND)
		set_3rdparty(pnetcdf EXTERNAL)
	else()
		set_3rdparty(pnetcdf INTERNAL)
	endif()
endif()

#------------------------------------------------------------------------------
#  APBS
#------------------------------------------------------------------------------

if(NEED_apbs)

	find_package(APBS)

	if(APBS_FOUND)
		set_3rdparty(apbs EXTERNAL)
	else()
		set_3rdparty(apbs DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  PUPIL
#------------------------------------------------------------------------------

if(NEED_pupil)
	find_package(PUPIL)
	if(PUPIL_FOUND)
		set_3rdparty(pupil EXTERNAL)
	else()
		set_3rdparty(pupil DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  LIO
#------------------------------------------------------------------------------

if(NEED_lio)

	find_package(LIO)

	if(LIO_FOUND)
		set_3rdparty(lio EXTERNAL)
	else()
		set_3rdparty(lio DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
# PLUMED
#------------------------------------------------------------------------------

if(NEED_plumed)

	# PLUMED changed its C API in version 2.5.
	# We can only build-time-link to versions >=2.5, though
	# runtime linking should work with all versions
	find_package(PLUMED 2.5)
	if(PLUMED_FOUND)
		set_3rdparty(plumed EXTERNAL)
	else()
		set_3rdparty(plumed DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  zlib, for cpptraj and netcdf
#------------------------------------------------------------------------------

if(NEED_zlib)
	find_package(ZLIB)

	if(ZLIB_FOUND)
		set_3rdparty(zlib EXTERNAL)
	else()
		set_3rdparty(zlib DISABLED)  # will always error
	endif()
endif()

#------------------------------------------------------------------------------
#  bzip2
#------------------------------------------------------------------------------
if(NEED_libbz2)
	find_package(BZip2)


	if(BZIP2_FOUND)
		set_3rdparty(libbz2 EXTERNAL)
	else()
		set_3rdparty(libbz2 DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  Math library
#------------------------------------------------------------------------------

if(NEED_libm)
	# figure out if we need a math library
	find_package(CMath)

	if(CMath_FOUND)
		set_3rdparty(libm EXTERNAL)
	else()
		set_3rdparty(libm DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  mpi4py (only needed for MMPBSA.py)
#------------------------------------------------------------------------------
if(NEED_mpi4py)
	check_python_package(mpi4py MPI4PY_FOUND)
	if(MPI4PY_FOUND)
		set_3rdparty(mpi4py EXTERNAL)
	else()

		if("${CMAKE_C_COMPILER_ID}" STREQUAL "PGI")
			message(STATUS "Cannot build internal mpi4py with PGI compiler")
			set_3rdparty(mpi4py DISABLED)
		else()
			set_3rdparty(mpi4py INTERNAL)
		endif()
	endif()
endif()

#------------------------------------------------------------------------------
#  PerlMol
#------------------------------------------------------------------------------
if(NEED_perlmol)

	if(BUILD_PERL)
		find_package(PerlModules COMPONENTS Chemistry::Mol ExtUtils::MakeMaker)

		if(EXISTS "${PERLMODULES_CHEMISTRY_MOL_MODULE}")
			set_3rdparty(perlmol EXTERNAL)
		else()
			if(HAVE_PERL_MAKE)

				if(EXISTS "${PERLMODULES_EXTUTILS_MAKEMAKER_MODULE}")
					set_3rdparty(perlmol INTERNAL)
				else()
					message(STATUS "Cannot build PerlMol internally because its dependency ExtUtils::MakeMaker is not installed.")
					set_3rdparty(perlmol DISABLED)
				endif()
			else()
				message(STATUS "Cannot build PerlMol internally because PERL_MAKE is not set to a valid program.")
				set_3rdparty(perlmol DISABLED)
			endif()
		endif()

	else()
		set_3rdparty(perlmol DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------
if(NEED_boost)

	set(Boost_DETAILED_FAILURE_MSG TRUE)
	find_package(Boost COMPONENTS thread system program_options iostreams regex timer chrono filesystem graph)

	set(EXT_BOOST_OK ${Boost_FOUND})

	if(EXT_BOOST_OK AND "${Boost_LIBRARIES}" STREQUAL "")
		message(WARNING "FindBoost reported success but returned no libraries to link to.  Please check your Boost installation and the output of FindBoost for details.")
		set(EXT_BOOST_OK FALSE)
	endif()

	if(EXT_BOOST_OK)
		check_boost_works(BOOST_WORKS)
		if(NOT BOOST_WORKS)
			set(EXT_BOOST_OK FALSE)
		endif()
	endif()

	if(EXT_BOOST_OK)
		check_boost_compression_support(BOOST_SUPPORTS_COMPRESSION)
		if(NOT BOOST_SUPPORTS_COMPRESSION)
			set(EXT_BOOST_OK FALSE)
		endif()
	endif()

	if(EXT_BOOST_OK)
		set_3rdparty(boost EXTERNAL)
	else()

		if(NOT libbz2_ENABLED AND zlib_ENABLED)
			message(STATUS "Unable to build internal Boost without zlib and libbz2")
			set_3rdparty(boost DISABLED)
		elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
			message(STATUS "PGI is not capable of compiling internal Boost.")
			set_3rdparty(boost DISABLED)
		endif()

		set_3rdparty(boost INTERNAL)
	endif()

endif()

#------------------------------------------------------------------------------
#  NVIDIA NCCL
#------------------------------------------------------------------------------

if(NEED_nccl)
	find_package(NCCL)

	if(NCCL_FOUND)
		set_3rdparty(nccl EXTERNAL)
	else()
		set_3rdparty(nccl DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  MBX
# (http://paesanigroup.ucsd.edu/software/mbx.html)
#------------------------------------------------------------------------------

if(NEED_mbx)
	find_package(MBX 0.2.3 CONFIG)

	if(MBX_FOUND)
		set_3rdparty(mbx EXTERNAL)
	else()
		message(STATUS "Could not find MBX.  To locate it, add its install dir to the prefix path.")
		set_3rdparty(mbx DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  TORCHANI
#------------------------------------------------------------------------------

if(NEED_torchani)
    find_package(Torchani 0.5 CONFIG)
    if(Torchani_FOUND)
        if (DOUBLE_PRECISION)
		    set_3rdparty(torchani EXTERNAL)
        else()
            message(STATUS "Can't link Torchani if DOUBLE_PRECISION is not enabled.")
		    set_3rdparty(torchani DISABLED)
        endif()
    else()
        message(STATUS "Could not find Torchani. To locate it, add its install dir to the prefix path.")
        set_3rdparty(torchani DISABLED)
    endif()
endif()

#------------------------------------------------------------------------------
#  tng_io
#------------------------------------------------------------------------------

if(NEED_tng_io)
	find_package(tng_io CONFIG)

	if(NOT tng_io_FOUND)
		message(STATUS "Could not find tng_io.  To locate it, add its install dir to the prefix path.")
	endif()

	if(tng_io_FOUND)
		set_3rdparty(tng_io EXTERNAL)
	elseif(zlib_ENABLED) # tng io requires zlib
		set_3rdparty(tng_io INTERNAL)
	else()
		set_3rdparty(tng_io DISABLED)
	endif()
endif()

#------------------------------------------------------------------------------
#  libtorch
#------------------------------------------------------------------------------

if(NEED_libtorch AND LIBTORCH)
	if (DEFINED TORCH_HOME)
		# use external libtorch
		if(NOT CUDA OR (CUDA AND CUDNN AND DEFINED CUDNN_INCLUDE_PATH AND DEFINED CUDNN_LIBRARY_PATH))
			set(CONTAIN_INSTALL_PREFIX FALSE)
			if(CMAKE_INSTALL_PREFIX)
				execute_process(COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/lib COMMAND_ERROR_IS_FATAL ANY)
				execute_process(COMMAND cp -r ${TORCH_HOME} ${CMAKE_INSTALL_PREFIX}/lib/libtorch COMMAND_ERROR_IS_FATAL ANY)
				set(CONTAIN_INSTALL_PREFIX TRUE)
			else()
				execute_process(COMMAND cp -r ${TORCH_HOME} ${CMAKE_CURRENT_BINARY_DIR}/libtorch COMMAND_ERROR_IS_FATAL ANY)
			endif()

			if(CONTAIN_INSTALL_PREFIX)
				set(TORCH_HOME ${CMAKE_INSTALL_PREFIX}/lib/libtorch)
			else()
				set(TORCH_HOME ${CMAKE_CURRENT_BINARY_DIR}/libtorch)
			endif()

			# edit cuda.cmake to config with amber and refind
			execute_process(COMMAND python ${CMAKE_SOURCE_DIR}/AmberTools/src/libtorch/fix_cuda.py ${TORCH_HOME}/share/cmake/Caffe2/public/cuda.cmake)

			# stack mkl from amber to protect libtorch env
			list(REMOVE_ITEM CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/hanjianwei")
			set(OLD_MKL_INCLUDE_DIR ${MKL_INCLUDE_DIR})
			set(OLD_MKL_LIBRARIES ${MKL_LIBRARIES})
			set(MKL_INCLUDE_DIR "")
			set(MKL_LIBRARIES "")

			# find libtorch
			find_package(Torch REQUIRED PATHS ${TORCH_HOME} NO_DEFAULT_PATH)

			# recover original mkl vars for further amber configuration
			set(MKL_INCLUDE_DIR ${OLD_MKL_INCLUDE_DIR})
			set(MKL_LIBRARIES ${OLD_MKL_LIBRARIES})
			list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/hanjianwei")

			message(STATUS "Libtorch: ${TORCH_HOME}")

			# set export variable
			set(libtorch_ENABLED TRUE CACHE INTERNAL "whether libtorch is enabled")
			set(LIBTORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS} CACHE INTERNAL "Libtorch header paths")
			set(LIBTORCH_LIBRARIES ${TORCH_LIBRARIES} CACHE INTERNAL "Libtorch library paths")

			set_3rdparty(libtorch EXTERNAL)
		else()
			message(FATAL_ERROR "Libtorch with CUDA requires CUDNN.  To find and configure it, define CUDNN, CUDNN_INCLUDE_PATH, and CUDNN_LIBRARY_PATH.")
		endif()
	else()
		# use internal libtorch
		set_3rdparty(libtorch INTERNAL)
	endif()
else()
	set(libtorch_ENABLED FALSE CACHE INTERNAL "whether libtorch is enabled")
	set_3rdparty(libtorch DISABLED)
endif()

#------------------------------------------------------------------------------
#  nlopt
#------------------------------------------------------------------------------

if(NEED_nlopt)
	#
	# the nlopt headers depend on the nlopt configuration
	# because we are relying on the internal headers, our
	# only option is to use the internal nlopt
	#
	#set_3rdparty(nlopt INTERNAL)

	#
	# In principle, we could search for external or use internal
	#
	find_package(nlopt)

	if(NOT nlopt_FOUND)
	  message(STATUS "Could not find nlopt.  To locate it, add its install dir to the prefix path.")
	endif()
	if(nlopt_FOUND)
	  set_3rdparty(nlopt EXTERNAL)
	  message("-- Found external nlopt")
	else()
	  set_3rdparty(nlopt INTERNAL)
	  message("-- Did not find external nlopt")
	endif()
endif()


if(NEED_xtb)
  #set(CMAKE_FIND_DEBUG_MODE TRUE)
  find_package(XTB REQUIRED)
  #set(CMAKE_FIND_DEBUG_MODE FALSE)
  if(XTB_FOUND)
    set_3rdparty(xtb EXTERNAL)
  else()
    set_3rdparty(xtb DISABLED)
  endif()
endif()

if(NEED_dftbplus)
  #set(CMAKE_FIND_DEBUG_MODE TRUE)
  find_package(DFTBPLUS CONFIG REQUIRED)
  #set(CMAKE_FIND_DEBUG_MODE FALSE)
  if(DFTBPLUS_FOUND)
    set_3rdparty(dftbplus EXTERNAL)
  else()
    set_3rdparty(dftbplus DISABLED)
  endif()
endif()

if(NEED_deepmd-kit)
	# J. Zeng: ideally, libdeepmd.tar.gz should contain a cmake config
	# However, it's not ready in the current version...
	# so I manually add FindDeePMD.cmake
	find_package(DeePMD REQUIRED)
	if(DeePMD_FOUND)
		set_3rdparty(deepmd-kit EXTERNAL)
	else()
		set_3rdparty(deepmd-kit DISABLED)
	endif()
endif()



# Apply overrides
# -------------------------------------------------------------------------------------------------------------------------------------------------------

# we can now reset this back to the default behavior -- targets will be made PIC as needed in the individual CMake scripts
unset(CMAKE_POSITION_INDEPENDENT_CODE)

set(FORCE_EXTERNAL_LIBS "" CACHE STRING "3rd party libraries to force using the system version of. Accepts a semicolon-separated list of library names from the 3rd Party Libraries section of the build report.")
set(FORCE_INTERNAL_LIBS "" CACHE STRING "3rd party libraries to force to build inside Amber. Accepts a semicolon-separated list of library names from the 3rd Party Libraries section of the build report.")
set(FORCE_DISABLE_LIBS "" CACHE STRING "3rd party libraries to force Amber to not use at all. Accepts a semicolon-separated list of library names from the 3rd Party Libraries section of the build report.")

# look for and handle suspicious tools
if(NOT TRUST_SYSTEM_LIBS)
	set(FOUND_SUSPICIOUS_TOOLS FALSE)
	set(SUSPICIOUS_TOOLS_MESSAGE "")

	foreach(TOOL ${SUSPICIOUS_3RDPARTY_TOOLS})
		if(${TOOL}_EXTERNAL)
			set(FOUND_SUSPICIOUS_TOOLS TRUE)
			set(SUSPICIOUS_TOOLS_MESSAGE "${SUSPICIOUS_TOOLS_MESSAGE} ${TOOL}")

			if("${TOOL}" IN_LIST BUNDLED_3RDPARTY_TOOLS)
				set_3rdparty(${TOOL} INTERNAL)
			else()
				set_3rdparty(${TOOL} DISABLED)
			endif()
		endif()
	endforeach()

	if(FOUND_SUSPICIOUS_TOOLS)
		message(STATUS "The following libraries were found on your system, but are not being used:${SUSPICIOUS_TOOLS_MESSAGE}")
		message(STATUS "This is because the Amber devs have frequently seen broken installs of these libraries causing trouble.")
		message(STATUS "To tell ${PROJECT_NAME} to link to these libraries, you can select individual ones using -DFORCE_EXTERNAL_LIBS=<names>.")
		message(STATUS "Alternatively, you can request that ${PROJECT_NAME} link to all libraries it finds using -DTRUST_SYSTEM_LIBS=TRUE.")
	endif()
endif()

foreach(TOOL ${FORCE_EXTERNAL_LIBS})
	colormsg(YELLOW "Forcing ${TOOL} to be sourced externally")

	string(TOLOWER ${TOOL} TOOL_LOWERCASE)
	list_contains(VALID_TOOL ${TOOL_LOWERCASE} ${3RDPARTY_TOOLS})

	if(NOT VALID_TOOL)
		message(FATAL_ERROR "${TOOL} is not a valid 3rd party library name.")
	endif()

	set_3rdparty(${TOOL_LOWERCASE} EXTERNAL)
endforeach()

if(INSIDE_AMBER)
	foreach(TOOL ${FORCE_INTERNAL_LIBS})
		colormsg(GREEN "Forcing ${TOOL} to be built internally")

		string(TOLOWER ${TOOL} TOOL_LOWERCASE)
		list_contains(VALID_TOOL ${TOOL_LOWERCASE} ${3RDPARTY_TOOLS})

		if(NOT VALID_TOOL)
			message(FATAL_ERROR "${TOOL} is not a valid 3rd party library name.")
		endif()

		set_3rdparty(${TOOL_LOWERCASE} INTERNAL)
	endforeach()
endif()

foreach(TOOL ${FORCE_DISABLE_LIBS})
	colormsg(HIRED "Forcing ${TOOL} to be disabled")

	string(TOLOWER ${TOOL} TOOL_LOWERCASE)
	list_contains(VALID_TOOL ${TOOL_LOWERCASE} ${3RDPARTY_TOOLS})

	if(NOT VALID_TOOL)
		message(FATAL_ERROR "${TOOL} is not a valid 3rd party library name.")
	endif()

	set_3rdparty(${TOOL_LOWERCASE} DISABLED)
endforeach()

# force all unneeded tools to be disabled
foreach(TOOL ${3RDPARTY_TOOLS})
	list(FIND NEEDED_3RDPARTY_TOOLS ${TOOL} TOOL_INDEX)

	if(${TOOL_INDEX} EQUAL -1)
		set_3rdparty(${TOOL} DISABLED)
	endif()

endforeach()

# check math library configuration
if(LINALG_LIBS_REQUIRED AND NOT (mkl_ENABLED OR (blas_ENABLED AND lapack_ENABLED)))
	message(FATAL_ERROR "You must enable a linear algebra library -- either blas and lapack, or mkl")
endif()

if(mkl_ENABLED AND (blas_ENABLED AND lapack_ENABLED))
	# prefer MKL to BLAS
	set_3rdparty(blas DISABLED)
	set_3rdparty(lapack DISABLED)
endif()

# Now that we know which libraries we need, set them up properly.
# -------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# c9xcomplex
#------------------------------------------------------------------------------

if(c9x-complex_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS c9x-complex)
endif()

#------------------------------------------------------------------------------
# check ucpp, import the system version
#------------------------------------------------------------------------------
if(ucpp_EXTERNAL)
	import_executable(ucpp ${UCPP_LOCATION})
elseif(ucpp_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS ucpp-1.3)
endif()

#------------------------------------------------------------------------------
#  Readline
#------------------------------------------------------------------------------

if(readline_EXTERNAL)

	# rename target
	import_libraries(readline
		LIBRARIES readline::readline
		INCLUDES ${READLINE_INCLUDE_DIR}/readline)
elseif(readline_INTERNAL)

	list(APPEND 3RDPARTY_SUBDIRS cpptraj/src/readline)

endif()

#------------------------------------------------------------------------------
#  MKL
#------------------------------------------------------------------------------


if(mkl_ENABLED)
	if(NOT MKL_FOUND)
		message(FATAL_ERROR "You enabled MKL, but it was not found.")
	endif()

	if(MIXING_COMPILERS AND OPENMP)
		message(WARNING "You are using different compilers from different vendors together.  This may cause link errors with MKL and OpenMP.  There is no way around this.")
	endif()

	if(mkl_ENABLED AND (blas_ENABLED OR lapack_ENABLED))
		message(FATAL_ERROR "MKL replaces blas and lapack!  They can't be enabled when MKL is in use!")
	endif()

	# add to library tracker
	import_libraries(mkl LIBRARIES ${MKL_FORTRAN_LIBRARIES} INCLUDES ${MKL_INCLUDE_DIRS})
endif()

#------------------------------------------------------------------------------
#  FFTW
#------------------------------------------------------------------------------

if(fftw_EXTERNAL)

	if(NOT FFTW_FOUND)
		message(FATAL_ERROR "Could not find FFTW, but it was set so be sourced externally!")
	endif()

	# rename targets
	import_libraries(fftw LIBRARIES fftw::fftwl)

	if(MPI)
		import_libraries(fftw_mpi LIBRARIES fftw::mpi_l)
	endif()

elseif(fftw_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS fftw-3.3)
endif()

#------------------------------------------------------------------------------
#  KMMD
#------------------------------------------------------------------------------

if(kmmd_EXTERNAL)

	if(NOT KMMD_FOUND)
		message(FATAL_ERROR "Could not find KMMD, but it was set so be sourced externally!")
	endif()

	# rename targets
	using_library_targets(kmmd::kmmd)
	import_libraries(kmmd LIBRARIES kmmd::kmmd)

elseif(kmmd_INTERNAL)

    message(STATUS "KMMD not added to 3RDPARTY_SUBDIRS :")
#- 	list(APPEND 3RDPARTY_SUBDIRS kmmd)
    message(STATUS "  : " "${3RDPARTY_SUBDIRS}")

endif()


#------------------------------------------------------------------------------
#  NetCDF
#------------------------------------------------------------------------------

if(netcdf_EXTERNAL)

	if(NOT NetCDF_FOUND)
		message(FATAL_ERROR "netcdf was set to be sourced externally, but it was not found!")
	endif()

	# Import the system netcdf as a library
	import_library(netcdf ${NetCDF_LIBRARIES_C} ${NetCDF_INCLUDES})

elseif(netcdf_INTERNAL)

	if(${COMPILER} STREQUAL CRAY)
			message(FATAL_ERROR "Bundled NetCDF cannot be used with cray compilers.  Please reconfigure with -DFORCE_EXTERNAL_LIBS=netcdf. \
		 On cray systems you can usually load the system NetCDF with 'module load cray-netcdf' or 'module load netcdf'.")
	endif()

	list(APPEND 3RDPARTY_SUBDIRS netcdf-4.6.1)
endif()

if(netcdf-fortran_EXTERNAL)

	if(NOT NetCDF_F90_FOUND)
		message(FATAL_ERROR "netcdf-fortran was set to be sourced externally, but it was not found!")
	endif()

	if(netcdf_INTERNAL)
		message(FATAL_ERROR "Cannot use internal netcdf with external netcdf-fortran!")
	endif()

	# Import the system netcdf as a library
	import_library(netcdff ${NetCDF_LIBRARIES_F90} ${NetCDF_INCLUDES_F90})
	set_property(TARGET netcdff PROPERTY INTERFACE_LINK_LIBRARIES netcdf)

	# This is really for symmetry with the other MOD_DIRs more than anything.
	set(NETCDF_FORTRAN_MOD_DIR ${NetCDF_INCLUDES_F90})

elseif(netcdf-fortran_INTERNAL)
	#TODO on Cray systems a static netcdf may be required

	if(${COMPILER} STREQUAL CRAY)
			message(FATAL_ERROR "Bundled NetCDF cannot be used with cray compilers. \
 On cray systems you can usually load the system NetCDF with 'module load cray-netcdf' or 'module load netcdf'.")
	endif()

	list(APPEND 3RDPARTY_SUBDIRS netcdf-fortran-4.4.4)
endif()

#------------------------------------------------------------------------------
#  Protobuf
#------------------------------------------------------------------------------

if(protobuf_EXTERNAL)

	if(NOT Protobuf_FOUND)
		message(FATAL_ERROR "protobuf was set to be sourced externally, but it was not found!")
	endif()

	# Import the system protobuf as a library
	import_library(protobuf ${Protobuf_LIBRARIES} ${Protobuf_INCLUDES})

elseif(protobuf_INTERNAL)

	#if(${COMPILER} STREQUAL CRAY)
	#	message(FATAL_ERROR "Bundled protobuf cannot be used with cray compilers.  Please reconfigure with -DFORCE_EXTERNAL_LIBS=protobuf.")
	#endif()

	list(APPEND 3RDPARTY_SUBDIRS protobuf-3.14.0/cmake)
endif()

#------------------------------------------------------------------------------
#  XBlas
#------------------------------------------------------------------------------

if(xblas_EXTERNAL)

	# rename target
	import_libraries(xblas LIBRARIES xblas::xblas)

elseif(xblas_INTERNAL)

	list(APPEND 3RDPARTY_SUBDIRS xblas)

endif()

#------------------------------------------------------------------------------
#  Netlib libraries
#------------------------------------------------------------------------------

# BLAS
if(blas_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS blas)
elseif(blas_EXTERNAL)
	import_libraries(blas LIBRARIES "${BLAS_LIBRARIES}")
endif()

#  LAPACK
if(lapack_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS lapack)
elseif(lapack_EXTERNAL)
	import_libraries(lapack LIBRARIES "${LAPACK_LIBRARIES}")
endif()


#  ARPACK
if(arpack_EXTERNAL)
	if(NOT EXISTS "${ARPACK_LIBRARY}")
		message(FATAL_ERROR "arpack was set to be sourced externally, but it was not found!")
	endif()

	import_library(arpack ${ARPACK_LIBRARY})

	if(NOT ARPACK_HAS_ARSECOND)
		message(STATUS "System arpack is missing the arsecond_ function.  That function will be built inside amber")
	endif()

elseif(arpack_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS arpack)
endif()

# --------------------------------------------------------------------
#  Parallel NetCDF
# --------------------------------------------------------------------

if(pnetcdf_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS pnetcdf)
elseif(pnetcdf_EXTERNAL)
	if(NOT PnetCDF_FOUND)
		message(FATAL_ERROR "You requested to use an external pnetcdf, but no installation was found.")
	endif()

	# rename target
	import_libraries(pnetcdf LIBRARIES pnetcdf::pnetcdf)

endif()

#------------------------------------------------------------------------------
#  APBS
#------------------------------------------------------------------------------

if(apbs_EXTERNAL)
	if(NOT APBS_FOUND)
		message(FATAL_ERROR "You requested to use external apbs, but no installation was found.")
	endif()

	import_libraries(apbs LIBRARIES ${APBS_LIBRARIES})
endif()

#------------------------------------------------------------------------------
#  PUPIL
#------------------------------------------------------------------------------

if(pupil_EXTERNAL)
	using_library_targets(pupil::pupil)
endif()

#------------------------------------------------------------------------------
#  LIO
#------------------------------------------------------------------------------

if(lio_EXTERNAL)
	using_library_targets(lio::lio)
endif()

#------------------------------------------------------------------------------
# PLUMED
#------------------------------------------------------------------------------
if(plumed_EXTERNAL)
	set(PLUMED_RUNTIME_LINK FALSE)
else()
	if(HAVE_LIBDL AND NEED_plumed)
		message(STATUS "Cannot find PLUMED.  You will still be able to load it at runtime.  If you want to link it at build time, set PLUMED_ROOT to where you installed it.")

		set(PLUMED_RUNTIME_LINK TRUE)
	else()
		set(PLUMED_RUNTIME_LINK FALSE)
	endif()
endif()

#------------------------------------------------------------------------------
# zlib
#------------------------------------------------------------------------------
if(zlib_EXTERNAL)
	using_library_targets(ZLIB::ZLIB)
endif()

#------------------------------------------------------------------------------
# libbz2
#------------------------------------------------------------------------------

if(libbz2_EXTERNAL)
	using_library_targets(BZip2::BZip2)
endif()

#------------------------------------------------------------------------------
#  mpi4py
#------------------------------------------------------------------------------

if(mpi4py_EXTERNAL)
	if(NOT MPI4PY_FOUND)
		message(FATAL_ERROR "mpi4py was set to be sourced externally, but the mpi4py package was not found.")
	endif()
elseif(mpi4py_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS mpi4py-3.1.4)
endif()

#------------------------------------------------------------------------------
#  perlmol
#------------------------------------------------------------------------------

if(perlmol_EXTERNAL)
	 if(NOT EXISTS "${PERLMODULES_CHEMISTRY_MOL_MODULE}")
		message(FATAL_ERROR "The Chemistry::Mol perl package was set to be sourced externally, but it was not found.")
	endif()
elseif(perlmol_INTERNAL)

	if(NOT HAVE_PERL_MAKE)
		message(FATAL_ERROR "A perl-compatible make program is required to build Chemistry::Mol")
	endif()
	list(APPEND 3RDPARTY_SUBDIRS PerlMol-0.3500)
endif()

#------------------------------------------------------------------------------
#  boost
#------------------------------------------------------------------------------

if(boost_EXTERNAL)
	if(NOT Boost_FOUND)
		message(FATAL_ERROR "boost was set to be sourced externally, but it was not found.")
	endif()

	set(BOOST_LIBS_TO_IMPORT chrono filesystem graph iostreams
	program_options regex system thread timer)

	foreach(BOOST_LIB ${BOOST_LIBS_TO_IMPORT})
		string(TOUPPER "${BOOST_LIB}" BOOST_LIB_UCASE)

		# the contents of ${Boost_${BOOST_LIB_UCASE}_LIBRARY} may be either the name
		# of an imported target, or a path to a library
		import_libraries(boost_${BOOST_LIB} LIBRARIES ${Boost_${BOOST_LIB_UCASE}_LIBRARY} INCLUDES ${Boost_INCLUDE_DIRS})

	endforeach()

	# header interface library
	add_library(boost_headers INTERFACE)
	set_property(TARGET boost_headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})


elseif(boost_INTERNAL)

	if(NOT libbz2_ENABLED AND zlib_ENABLED)
		message(FATAL_ERROR "Unable to build internal Boost without zlib and libbz2.")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
		message(FATAL_ERROR "PGI is not capable of compiling internal Boost.  An external Boost must be supplied.")
	endif()


	list(APPEND 3RDPARTY_SUBDIRS boost)

endif()

#------------------------------------------------------------------------------
#  NCCL
#------------------------------------------------------------------------------

if(nccl_EXTERNAL)
	using_library_targets(nccl)
endif()

#------------------------------------------------------------------------------
#  mbx
#------------------------------------------------------------------------------

if(mbx_EXTERNAL)
	using_library_targets(MBX::mbx)
endif()

#------------------------------------------------------------------------------
#  Torchani
#------------------------------------------------------------------------------

if(torchani_EXTERNAL)
    using_library_targets(Torchani::torchani)
endif()

# --------------------------------------------------------------------
#  tng_io
# --------------------------------------------------------------------

if(tng_io_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS cpptraj/src/tng)
elseif(tng_io_EXTERNAL)
	if(NOT tng_io_FOUND)
		message(FATAL_ERROR "You requested to use an external libtng_io, but no installation was found.")
	endif()

	# rename target
	import_libraries(tng_io LIBRARIES tng_io::tng_io)
endif()

#------------------------------------------------------------------------------
# libtorch
#------------------------------------------------------------------------------

if(libtorch_INTERNAL)
	list(APPEND 3RDPARTY_SUBDIRS libtorch)
elseif(libtorch_EXTERNAL)
	if(NOT Torch_FOUND)
		message(FATAL_ERROR "You requested to use an external libtorch, but no installation was found.")
	endif()
endif()

#------------------------------------------------------------------------------
#  nlopt
#------------------------------------------------------------------------------ 

if(nlopt_INTERNAL)
  list(APPEND 3RDPARTY_SUBDIRS nlopt)
elseif(nlopt_EXTERNAL)
  if(NOT nlopt_FOUND)
    message(FATAL_ERROR "You requested to use an external nlopt, but no installation was found.")
  endif()
  
  #using_library_targets(nlopt LIBRARIES nlopt::nlopt)
  #using_library_targets(nlopt::nlopt)
  #import_libraries(nlopt LIBRARIES nlopt::nlopt INCLUDES nlopt::nlopt)
  #using_library_targets(nlopt LIBRARIES nlopt::nlopt INCLUDES nlopt::nlopt)
  using_library_targets(nlopt::nlopt)
  import_libraries(nlopt LIBRARIES nlopt::nlopt)
endif()
