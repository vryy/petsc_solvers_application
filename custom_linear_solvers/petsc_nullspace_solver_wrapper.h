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
template<class TSparseSpaceType, class TDenseSpaceType>
class PetscNullspaceSolverWrapper : public LinearSolver<TSparseSpaceType, TDenseSpaceType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscNullspaceSolverWrapper);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    typedef typename TSparseSpaceType::ValueType ValueType;

    /**
     * Default Constructor
     */
    PetscNullspaceSolverWrapper(typename BaseType::Pointer pLinearSolver) : m_my_rank(0), mpLinearSolver(pLinearSolver)
    {
    }

    /**
     * Destructor
     */
    virtual ~PetscNullspaceSolverWrapper()
    {
    }

    /** Some solvers may require a minimum degree of knowledge of the structure of the matrix. To make an example
     * when solving a mixed u-p problem, it is important to identify the row associated to v and p.
     * another example is the automatic prescription of rotation null-space for smoothed-aggregation solvers
     * which require knowledge on the spatial position of the nodes associated to a given dof.
     * This function tells if the solver requires such data
     */
    virtual bool AdditionalPhysicalDataIsNeeded()
    {
        return true;
    }

    /** Some solvers may require a minimum degree of knowledge of the structure of the matrix. To make an example
     * when solving a mixed u-p problem, it is important to identify the row associated to v and p.
     * another example is the automatic prescription of rotation null-space for smoothed-aggregation solvers
     * which require knowledge on the spatial position of the nodes associated to a given dof.
     * This function is the place to eventually provide such data
     */
    virtual void ProvideAdditionalData(
        SparseMatrixType& rA,
        VectorType& rX,
        VectorType& rB,
        typename ModelPart::DofsArrayType& rdof_set,
        ModelPart& r_model_part
    )
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
        typename ModelPart::NodesContainerType nodes = r_model_part.Nodes();
        for(typename ModelPart::DofsArrayType::iterator dof_iterator = rdof_set.begin();
                dof_iterator != rdof_set.end(); ++dof_iterator)
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
    virtual bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB)
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
    virtual bool Solve(SparseMatrixType& rA, DenseMatrixType& rX, DenseMatrixType& rB)
    {
        KRATOS_THROW_ERROR(std::logic_error, "Not yet implement", __FUNCTION__)
    }

    /**
     * Print information about this object.
     */
    virtual void PrintInfo(std::ostream& rOStream) const
    {
        mpLinearSolver->PrintInfo(rOStream);
        if(m_my_rank == 0)
            rOStream << "Petsc nullspace solver wrapper finished.";
    }

    /**
     * Print object's data.
     */
    virtual void PrintData(std::ostream& rOStream) const
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


/**
 * input stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType,class TReordererType>
inline std::istream& operator >> (std::istream& rIStream, PetscNullspaceSolverWrapper<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    return rIStream;
}

/**
 * output stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream, const PetscNullspaceSolverWrapper<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}


}  // namespace Kratos.

#undef DEBUG_SOLVER
#undef TEST_NULLSPACE

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_NULLSPACE_SOLVER_WRAPPER_H_INCLUDED  defined 

