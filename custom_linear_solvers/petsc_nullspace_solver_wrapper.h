/*
see petsc_solvers_application/LICENSE.txt
*/

//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: 29 May 2017 $
//   Revision:            $Revision: 1.1 $
//
//

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_NULLSPACE_SOLVER_WRAPPER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_NULLSPACE_SOLVER_WRAPPER_H_INCLUDED

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
#define TEST_NULLSPACE

namespace Kratos
{

/*
Solver wrapper to set the near nullspace to the matrix. This is particular useful for 3D elasticity and GAMG.
*/
template<class TSparseSpaceType, class TDenseSpaceType, class TModelPartType>
class PetscNullspaceSolverWrapper : public LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscNullspaceSolverWrapper);

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
    PetscNullspaceSolverWrapper(typename BaseType::Pointer pLinearSolver)
    : BaseType(), m_my_rank(0), mpLinearSolver(pLinearSolver)
    {
    }

    /**
     * Destructor
     */
    ~PetscNullspaceSolverWrapper() override
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

        PetscErrorCode  ierr;
        MatNullSpace    matnull;
        Vec             vec_coords;
        PetscScalar     *c;

        ierr = VecCreate(Comm, &vec_coords); //CHKERRQ(ierr);
////        ierr = VecSetBlockSize(vec_coords, 3); //CHKERRQ(ierr);
        ierr = VecSetSizes(vec_coords, static_cast<PetscInt>(rdof_set.size()), PETSC_DECIDE); //CHKERRQ(ierr);
        ierr = VecSetType(vec_coords, VECMPI);
//        ierr = VecSetUp(vec_coords); //CHKERRQ(ierr);
        ierr = VecGetArray(vec_coords, &c); //CHKERRQ(ierr);

        std::size_t i = 0, node_id;
        typename ModelPartType::NodesContainerType nodes = r_model_part.Nodes();
        for(auto dof_iterator = rdof_set.begin(); dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            node_id = dof_iterator->Id();
            if(dof_iterator->GetVariable() == DISPLACEMENT_X)
            {
                c[i++] = nodes[node_id].X();
            }
            else if(dof_iterator->GetVariable() == DISPLACEMENT_Y)
            {
                c[i++] = nodes[node_id].Y();
            }
            else if(dof_iterator->GetVariable() == DISPLACEMENT_Z)
            {
                c[i++] = nodes[node_id].Z();
            }
        }

        ierr = VecRestoreArray(vec_coords, &c); //CHKERRQ(ierr);
        ierr = MatNullSpaceCreateRigidBody(vec_coords, &matnull); //CHKERRQ(ierr);

        #ifdef TEST_NULLSPACE
        // test if the computed nullspace is really the nullspace of the system
        PetscBool isNull;
        MatNullSpaceTest(matnull, rA.Get(), &isNull);
        if(m_my_rank == 0)
        {
            if(!isNull)
                std::cout << "The computed nullspace does not pass MatNullSpaceTest" << std::endl;
            else
                std::cout << "The computed nullspace passes MatNullSpaceTest" << std::endl;
        }
        #endif

        ierr = MatSetNearNullSpace(rA.Get(), matnull); //CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&matnull); //CHKERRQ(ierr);
        ierr = VecDestroy(&vec_coords); //CHKERRQ(ierr);
        if(m_my_rank == 0)
            std::cout << "PetscNullspaceSolverWrapper::" << __FUNCTION__ << ", the near nullspace is set" << std::endl;

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
        bool solve_flag = mpLinearSolver->Solve(rA, rX, rB);

        return solve_flag;
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
            rOStream << "Petsc nullspace solver wrapper finished.";
    }

    /**
     * Print object's data.
     */
    void PrintData(std::ostream& rOStream) const override
    {
        mpLinearSolver->PrintData(rOStream);
    }

protected:
    int m_my_rank;

private:

    typename BaseType::Pointer mpLinearSolver;

    /**
     * Assignment operator.
     */
    PetscNullspaceSolverWrapper& operator=(const PetscNullspaceSolverWrapper& Other);
};

}  // namespace Kratos.

#undef DEBUG_SOLVER
#undef TEST_NULLSPACE

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_NULLSPACE_SOLVER_WRAPPER_H_INCLUDED  defined
