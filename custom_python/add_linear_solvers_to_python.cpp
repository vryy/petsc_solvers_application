//   
//   Project Name:        Kratos       
//   Last modified by:    $Author: hbui $
//   Date:                $Date: Apr 19, 2012 $
//   Revision:            $Revision: 1.0 $
//
//


// System includes 

// External includes 
#include <boost/python.hpp>

// Project includes
#include "includes/define.h"
#include "spaces/ublas_space.h"
#include "linear_solvers/linear_solver.h"
#include "custom_linear_solvers/petsc_solver.h"
#include "custom_linear_solvers/petsc_gamg_elasticity_solver.h"
#include "custom_linear_solvers/petsc_fieldsplit_u_wp_solver.h"
#include "custom_linear_solvers/petsc_fieldsplit_u_lu_solver.h"
#include "custom_linear_solvers/petsc_fieldsplit_ux_uy_uz_solver.h"
#include "custom_linear_solvers/petsc_fieldsplit_ux_uy_uz_shield_solver.h"
#include "custom_linear_solvers/petsc_fieldsplit_ux_uy_uz_wp_solver.h"
#include "custom_linear_solvers/petsc_fieldsplit_u_shield_solver.h"
#include "custom_linear_solvers/petsc_fieldsplit_u_shield_wp_solver.h"
#include "custom_linear_solvers/petsc_fieldsplit_u_nested_shield_wp_solver.h"
#include "custom_linear_solvers/petsc_scaling_wrapper.h"
#include "custom_linear_solvers/petsc_u_wp_scaling_wrapper.h"
#include "distributed_builders_application/custom_utilities/PETSc_Wrappers.h"
#include "distributed_builders_application/custom_spaces/petsc_space.h"


namespace Kratos
{

namespace Python
{
    void  PetscSolversApplication_AddLinearSolversToPython()
    {
        typedef UblasSpace<double, CompressedMatrix, Vector> LocalSparseSpaceType;
        typedef UblasSpace<double, Matrix, Vector> LocalSpaceType;

        typedef PETScSpace PETScSparseSpaceType;
        typedef UblasSpace<double, Matrix, Vector> PETScLocalSpaceType;

        using namespace boost::python;

        //***************************************************************************
        //linear solvers
        //***************************************************************************

        typedef LinearSolver<PETScSparseSpaceType, PETScLocalSpaceType> PETScLinearSolverType;
        class_<PETScLinearSolverType, PETScLinearSolverType::Pointer, boost::noncopyable>
        ("PETScLinearSolver", init<>())
        .def("AdditionalPhysicalDataIsNeeded", &PETScLinearSolverType::AdditionalPhysicalDataIsNeeded)
        .def("ProvideAdditionalData", &PETScLinearSolverType::ProvideAdditionalData)
        .def(self_ns::str(self))
        ;

        typedef PetscSolver<PETScSparseSpaceType, PETScLocalSpaceType> PetscSolverType;
        class_<PetscSolverType, PetscSolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScSolver", init<>())
        .def(self_ns::str(self))
        ;

        typedef PetscGAMGElasticitySolver<PETScSparseSpaceType, PETScLocalSpaceType> PetscGAMGElasticitySolverType;
        class_<PetscGAMGElasticitySolverType, PetscGAMGElasticitySolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScGAMGElasticitySolver", init<>())
        .def("SetWorkingDimension", &PetscGAMGElasticitySolverType::SetWorkingDimension)
        .def(self_ns::str(self))
        ;

        typedef PetscFieldSplit_U_WP_Solver<PETScSparseSpaceType, PETScLocalSpaceType> PetscFieldSplit_U_WP_SolverType;
        class_<PetscFieldSplit_U_WP_SolverType, PetscFieldSplit_U_WP_SolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScFieldSplit_U_WP_Solver", init<>())
        .def(init<bool>())
        .def(self_ns::str(self))
        ;

        typedef PetscFieldSplit_U_LU_Solver<PETScSparseSpaceType, PETScLocalSpaceType> PetscFieldSplit_U_LU_SolverType;
        class_<PetscFieldSplit_U_LU_SolverType, PetscFieldSplit_U_LU_SolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScFieldSplit_U_LU_Solver", init<>())
        .def(init<bool>())
        .def(self_ns::str(self))
        ;

        typedef PetscFieldSplit_UX_UY_UZ_WP_Solver<PETScSparseSpaceType, PETScLocalSpaceType> PetscFieldSplit_UX_UY_UZ_WP_SolverType;
        class_<PetscFieldSplit_UX_UY_UZ_WP_SolverType, PetscFieldSplit_UX_UY_UZ_WP_SolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScFieldSplit_UX_UY_UZ_WP_Solver", init<>())
        .def(self_ns::str(self))
        ;

        typedef PetscFieldSplit_UX_UY_UZ_Solver<PETScSparseSpaceType, PETScLocalSpaceType> PetscFieldSplit_UX_UY_UZ_SolverType;
        class_<PetscFieldSplit_UX_UY_UZ_SolverType, PetscFieldSplit_UX_UY_UZ_SolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScFieldSplit_UX_UY_UZ_Solver", init<>())
        .def(self_ns::str(self))
        ;

        typedef PetscFieldSplit_UX_UY_UZ_Shield_Solver<PETScSparseSpaceType, PETScLocalSpaceType> PetscFieldSplit_UX_UY_UZ_Shield_SolverType;
        class_<PetscFieldSplit_UX_UY_UZ_Shield_SolverType, PetscFieldSplit_UX_UY_UZ_Shield_SolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScFieldSplit_UX_UY_UZ_Shield_Solver", init<boost::python::list&>())
        .def(self_ns::str(self))
        ;

        typedef PetscFieldSplit_U_Shield_Solver<PETScSparseSpaceType, PETScLocalSpaceType> PetscFieldSplit_U_Shield_SolverType;
        class_<PetscFieldSplit_U_Shield_SolverType, PetscFieldSplit_U_Shield_SolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScFieldSplit_U_Shield_Solver", init<boost::python::list&>())
        .def(init<boost::python::list&, bool>())
        .def(self_ns::str(self))
        ;

        typedef PetscFieldSplit_U_Shield_WP_Solver<PETScSparseSpaceType, PETScLocalSpaceType> PetscFieldSplit_U_Shield_WP_SolverType;
        class_<PetscFieldSplit_U_Shield_WP_SolverType, PetscFieldSplit_U_Shield_WP_SolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScFieldSplit_U_Shield_WP_Solver", init<boost::python::list&>())
        .def(init<boost::python::list&, bool>())
        .def(self_ns::str(self))
        ;

        typedef PetscFieldSplit_U_Nested_Shield_WP_Solver<PETScSparseSpaceType, PETScLocalSpaceType> PetscFieldSplit_U_Nested_Shield_WP_SolverType;
        class_<PetscFieldSplit_U_Nested_Shield_WP_SolverType, PetscFieldSplit_U_Nested_Shield_WP_SolverType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScFieldSplit_U_Nested_Shield_WP_Solver", init<boost::python::list&>())
        .def(self_ns::str(self))
        ;

        typedef PetscScalingWrapper<PETScSparseSpaceType, PETScLocalSpaceType> PetscScalingWrapperType;
        class_<PetscScalingWrapperType, PetscScalingWrapperType::Pointer, bases<PETScLinearSolverType>, boost::noncopyable>
        ("PETScScalingWrapper", init<PETScLinearSolverType::Pointer>())
        .def(self_ns::str(self))
        ;

        typedef Petsc_U_WP_ScalingWrapper<PETScSparseSpaceType, PETScLocalSpaceType> Petsc_U_WP_ScalingWrapperType;
        class_<Petsc_U_WP_ScalingWrapperType, Petsc_U_WP_ScalingWrapperType::Pointer, bases<PetscScalingWrapperType>, boost::noncopyable>
        ("PETSc_U_WP_ScalingWrapper", init<PETScLinearSolverType::Pointer>())
        .def(self_ns::str(self))
        ;

    }

}  // namespace Python.

} // Namespace Kratos

