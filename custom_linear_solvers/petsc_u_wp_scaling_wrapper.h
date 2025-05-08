/*
see petsc_solvers_application/LICENSE.txt
*/

//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: 15 Jan 2016 $
//   Revision:            $Revision: 1.1 $
//
//

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_U_WP_SCALING_WRAPPER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_U_WP_SCALING_WRAPPER_H_INCLUDED

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>

// External includes
#include "boost/smart_ptr.hpp"
#include "petscksp.h"

// Project includes
#include "includes/define.h"
#include "includes/ublas_interface.h"
#include "utilities/openmp_utils.h"
#include "linear_solvers/linear_solver.h"
#include "petsc_scaling_wrapper.h"
#include "distributed_builders_application/distributed_builders_application.h" // for SCALING_* variables

#define DEBUG_SOLVER

namespace Kratos
{

template<class TSparseSpaceType, class TDenseSpaceType, class TModelPartType>
class Petsc_U_WP_ScalingWrapper : public PetscScalingWrapper<TSparseSpaceType, TDenseSpaceType, TModelPartType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(Petsc_U_WP_ScalingWrapper);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType> LinearSolverType;

    typedef PetscScalingWrapper<TSparseSpaceType, TDenseSpaceType, TModelPartType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    typedef typename TSparseSpaceType::ValueType ValueType;

    typedef typename BaseType::ModelPartType ModelPartType;

    /**
     * Default Constructor
     */
    Petsc_U_WP_ScalingWrapper(typename LinearSolverType::Pointer pLinearSolver) : BaseType(pLinearSolver)
    {
    }

    /**
     * Destructor
     */
    ~Petsc_U_WP_ScalingWrapper() override
    {
    }

    /** Some solvers may require a minimum degree of knowledge of the structure of the matrix. To make an example
     * when solving a mixed u-p problem, it is important to identify the row associated to v and p.
     * another example is the automatic prescription of rotation null-space for smoothed-aggregation solvers
     * which require knowledge on the spatial position of the nodes associated to a given dof.
     * This function tells if the solver requires such data
     */
    bool AdditionalPhysicalDataIsNeeded() override
    {
        return true;
    }

    /** Some solvers may require a minimum degree of knowledge of the structure of the matrix. To make an example
     * when solving a mixed u-p problem, it is important to identify the row associated to v and p.
     * another example is the automatic prescription of rotation null-space for smoothed-aggregation solvers
     * which require knowledge on the spatial position of the nodes associated to a given dof.
     * This function is the place to eventually provide such data
     */
    void ProvideAdditionalData(
        SparseMatrixType& rA,
        VectorType& rX,
        VectorType& rB,
        typename ModelPartType::DofsArrayType& rdof_set,
        ModelPartType& r_model_part
    ) override
    {
        IndexType       Istart, Iend;
        PetscErrorCode  ierr;

        // construct the scaling vects
        TSparseSpaceType::Duplicate(rX, BaseType::mRightScalingVect);

        ierr = MatGetOwnershipRange(rA.Get(), &Istart, &Iend); CHKERRV(ierr);
//        KRATOS_WATCH(Istart)
//        KRATOS_WATCH(Iend)

        TSparseSpaceType::AssembleBegin(BaseType::mRightScalingVect);

        IndexType row;
        ValueType val;
        for(typename ModelPart::DofsArrayType::iterator dof_iterator = rdof_set.begin();
                dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            row = dof_iterator->EquationId();
            if((row >= Istart) && (row < Iend))
            {
                if(dof_iterator->GetVariable() == DISPLACEMENT_X)
                {
                    val = dof_iterator->GetSolutionStepValue(SCALING_FACTOR_DISPLACEMENT_X);
                    VecSetValue(BaseType::mRightScalingVect.Get(), row, val, INSERT_VALUES);
                }
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Y)
                {
                    val = dof_iterator->GetSolutionStepValue(SCALING_FACTOR_DISPLACEMENT_Y);
                    VecSetValue(BaseType::mRightScalingVect.Get(), row, val, INSERT_VALUES);
                }
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Z)
                {
                    val = dof_iterator->GetSolutionStepValue(SCALING_FACTOR_DISPLACEMENT_Z);
                    VecSetValue(BaseType::mRightScalingVect.Get(), row, val, INSERT_VALUES);
                }
                else if(dof_iterator->GetVariable() == WATER_PRESSURE)
                {
                    val = dof_iterator->GetSolutionStepValue(SCALING_FACTOR_WATER_PRESSURE);
                    VecSetValue(BaseType::mRightScalingVect.Get(), row, val, INSERT_VALUES);
                }
            }
        }

        TSparseSpaceType::AssembleEnd(BaseType::mRightScalingVect);

        BaseType::ProvideAdditionalData(rA, rX, rB, rdof_set, r_model_part);
    }

    /**
     * Print information about this object.
     */
    void PrintInfo(std::ostream& rOStream) const override
    {
        BaseType::PrintInfo(rOStream);
        if(BaseType::m_my_rank == 0)
            rOStream << "Petsc U-WP scaling wrapper finished.";
    }

    /**
     * Print object's data.
     */
    void PrintData(std::ostream& rOStream) const override
    {
        BaseType::PrintData(rOStream);
    }

private:

    /**
     * Assignment operator.
     */
    Petsc_U_WP_ScalingWrapper& operator=(const Petsc_U_WP_ScalingWrapper& Other);
};

}  // namespace Kratos.

#undef DEBUG_SOLVER

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_SCALING_WRAPPER_H_INCLUDED  defined
