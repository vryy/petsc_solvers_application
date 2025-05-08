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

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_SCALING_WRAPPER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_SCALING_WRAPPER_H_INCLUDED

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

#define DEBUG_SOLVER

namespace Kratos
{

template<class TSparseSpaceType, class TDenseSpaceType, class TModelPartType>
class PetscScalingWrapper : public LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscScalingWrapper);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    typedef typename TSparseSpaceType::ValueType ValueType;

    typedef typename BaseType::ModelPartType ModelPartType;

    /**
     * Default Constructor
     */
    PetscScalingWrapper(typename BaseType::Pointer pLinearSolver)
    : BaseType(), m_my_rank(0), mpLinearSolver(pLinearSolver)
    {
    }

    /**
     * Destructor
     */
    ~PetscScalingWrapper() override
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
        MPI_Comm Comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));
        MPI_Comm_rank(Comm, &m_my_rank);

        MatDiagonalScale(rA.Get(), mLeftScalingVect.Get(), mRightScalingVect.Get());
        if(!mLeftScalingVect.IsNULL())
            VecPointwiseMult(rB.Get(), mLeftScalingVect.Get(), rB.Get());

        if(mpLinearSolver->AdditionalPhysicalDataIsNeeded())
            mpLinearSolver->ProvideAdditionalData(rA, rX, rB, rdof_set, r_model_part);
    }

    /**
     * Normal solve method.
     * Solves the linear system Ax=b and puts the result on SystemVector& rX.
     * rX is also th initial guess for iterative methods.
     * @param rA. System matrix
     * @param rX. Solution vector.
     * @param rB. Right hand side vector.
     */
    bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB) override
    {
        bool solved = mpLinearSolver->Solve(rA, rX, rB);

        // rescale back the solution
        if(!mRightScalingVect.IsNULL())
            VecPointwiseMult(rX.Get(), mRightScalingVect.Get(), rX.Get());

        return solved;
    }

    /**
     * Multi solve method for solving a set of linear systems with same coefficient matrix.
     * Solves the linear system Ax=b and puts the result on SystemVector& rX.
     * rX is also th initial guess for iterative methods.
     * @param rA. System matrix
     * @param rX. Solution vector.
     * @param rB. Right hand side vector.
     */
    bool Solve(SparseMatrixType& rA, DenseMatrixType& rX, DenseMatrixType& rB) override
    {
        KRATOS_ERROR << "Not yet implement";
    }

    /**
     * Print information about this object.
     */
    void PrintInfo(std::ostream& rOStream) const override
    {
        mpLinearSolver->PrintInfo(rOStream);
        if(m_my_rank == 0)
            rOStream << "Petsc scaling wrapper finished.";
    }

    /**
     * Print object's data.
     */
    void PrintData(std::ostream& rOStream) const override
    {
        mpLinearSolver->PrintData(rOStream);
    }

protected:
    VectorType mLeftScalingVect;
    VectorType mRightScalingVect;
    int m_my_rank;

private:

    typename BaseType::Pointer mpLinearSolver;

    /**
     * Assignment operator.
     */
    PetscScalingWrapper& operator=(const PetscScalingWrapper& Other);
};

}  // namespace Kratos.

#undef DEBUG_SOLVER

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_SCALING_WRAPPER_H_INCLUDED  defined
