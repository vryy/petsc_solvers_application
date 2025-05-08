/*
see petsc_solvers_application/LICENSE.txt
*/

//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: 14 Jan 2016 $
//   Revision:            $Revision: 1.0 $
//
//

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_WP_SOLVER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_WP_SOLVER_H_INCLUDED

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
#define APPLY_NEAR_NULLSPACE
#define APPLY_COORDINATES

namespace Kratos
{

/**
This class constructs Petsc solver with PCFIELDSPLIT preconditioner.
This class assumes the provided DofSet is organized with contiguous tuple {ux,u_y,u_z} and the water pressure is single field. Hence it shall only be used with the parallel block builder and solver.
*/
template<class TSparseSpaceType, class TDenseSpaceType, class TModelPartType>
class PetscFieldSplit_U_WP_Solver : public LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscFieldSplit_U_WP_Solver);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    typedef typename BaseType::ModelPartType ModelPartType;

    /**
     * Default Constructor
     */
    PetscFieldSplit_U_WP_Solver()
    : BaseType(), m_my_rank(0), m_is_block(true)
    {
    }

    PetscFieldSplit_U_WP_Solver(bool is_block)
    : BaseType(), m_my_rank(0), m_is_block(is_block)
    {
    }

    /**
     * Destructor
     */
    ~PetscFieldSplit_U_WP_Solver() override
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
        // TODO collect the equation id for displacements and water pressure in the local process
        IndexType       Istart, Iend;
        MPI_Comm        Comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));
        PetscErrorCode  ierr;

        MPI_Comm_rank(Comm, &m_my_rank);

        ierr = MatGetOwnershipRange(rA.Get(), &Istart, &Iend); CHKERRV(ierr);
//        KRATOS_WATCH(Istart)
//        KRATOS_WATCH(Iend)

        mIndexU.clear();
        mIndexWP.clear();
        for(auto dof_iterator = rdof_set.begin();
                dof_iterator != rdof_set.end(); ++dof_iterator)
        {
//            std::size_t node_id = dof_iterator->Id();
            std::size_t row_id = dof_iterator->EquationId();

            if((row_id >= Istart) && (row_id < Iend))
            {
//                typename ModelPartType::NodesContainerType::iterator i_node = r_model_part.Nodes().find(node_id);
//                if(i_node == r_model_part.Nodes().end())
//                    KRATOS_ERROR << "The node does not exist in this partition. Probably data is consistent";

                if(dof_iterator->GetVariable() == DISPLACEMENT_X)
                    mIndexU.push_back(row_id);
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Y)
                    mIndexU.push_back(row_id);
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Z)
                    mIndexU.push_back(row_id);
                else if(dof_iterator->GetVariable() == WATER_PRESSURE)
                    mIndexWP.push_back(row_id);
            }
        }

        #if defined(APPLY_NEAR_NULLSPACE) || defined(APPLY_COORDINATES)
        std::size_t cnt, node_id;
        typename ModelPartType::NodesContainerType nodes = r_model_part.Nodes();
        #endif

        #ifdef APPLY_COORDINATES
        if(mcoords.size() != rdof_set.size())
            mcoords.resize(rdof_set.size());
        cnt = 0;
        for(auto dof_iterator = rdof_set.begin(); dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            node_id = dof_iterator->Id();
            if(dof_iterator->GetVariable() == DISPLACEMENT_X)
            {
                mcoords[cnt++] = nodes[node_id].X();
            }
            else if(dof_iterator->GetVariable() == DISPLACEMENT_Y)
            {
                mcoords[cnt++] = nodes[node_id].Y();
            }
            else if(dof_iterator->GetVariable() == DISPLACEMENT_Z)
            {
                mcoords[cnt++] = nodes[node_id].Z();
            }
        }
        mcoords.resize(cnt);
        #else
        cnt = 0;
        for(auto dof_iterator = rdof_set.begin(); dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            if(dof_iterator->GetVariable() == DISPLACEMENT_X) {cnt++;}
            else if(dof_iterator->GetVariable() == DISPLACEMENT_Y) {cnt++;}
            else if(dof_iterator->GetVariable() == DISPLACEMENT_Z) {cnt++;}
        }
        #endif

        // make sure that the number of displacement dofs is divisible to 3
        if (cnt % 3 != 0)
        {
            KRATOS_ERROR << "The number of displacement dofs is not divisible by 3 at process rank " << m_my_rank;
        }

        #ifdef APPLY_NEAR_NULLSPACE
        // construct the near nullspace of the solid block if required
        Vec             vec_coords;
        PetscScalar     *c;

        ierr = VecCreate(Comm, &vec_coords); //CHKERRQ(ierr);
        ierr = VecSetBlockSize(vec_coords, 3); //CHKERRQ(ierr);
        ierr = VecSetSizes(vec_coords, static_cast<PetscInt>(cnt), PETSC_DECIDE); //CHKERRQ(ierr);
        ierr = VecSetType(vec_coords, VECMPI);
//        ierr = VecSetUp(vec_coords); //CHKERRQ(ierr);
        ierr = VecGetArray(vec_coords, &c); //CHKERRQ(ierr);

        cnt = 0;
        for(auto dof_iterator = rdof_set.begin(); dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            node_id = dof_iterator->Id();
            if(dof_iterator->GetVariable() == DISPLACEMENT_X)
            {
                c[cnt++] = nodes[node_id].X();
            }
            else if(dof_iterator->GetVariable() == DISPLACEMENT_Y)
            {
                c[cnt++] = nodes[node_id].Y();
            }
            else if(dof_iterator->GetVariable() == DISPLACEMENT_Z)
            {
                c[cnt++] = nodes[node_id].Z();
            }
        }

        ierr = VecRestoreArray(vec_coords, &c); //CHKERRQ(ierr);
        ierr = MatNullSpaceCreateRigidBody(vec_coords, &mmatnull); //CHKERRQ(ierr);
        ierr = VecDestroy(&vec_coords); //CHKERRQ(ierr);
        #endif
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
        Vec             r;             /* approx solution, RHS, A*x-b */
        KSP             ksp;               /* linear solver context */
        PC              pc;           /* preconditioner context */
        IS              IS_u, IS_wp;   /* index set context */
        PetscReal       norm_b, norm_r;     /* ||b||, ||b-Ax|| */
        IndexType       its;
        PetscErrorCode  ierr;
        PetscScalar     v;
        MPI_Comm        Comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));

        MPI_Comm_rank(Comm, &m_my_rank);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /*
            Create linear solver context
        */
        ierr = KSPCreate(Comm, &ksp); CHKERRQ(ierr);
        if(ierr != 0)
            KRATOS_ERROR << "Error at KSPCreate, error code = " << ierr;

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
            std::cout << "KSPCreate completed" << std::endl;
        #endif

        /*
            Set operators. Here the matrix that defines the linear system
            also serves as the preconditioning matrix.
        */
        ierr = KSPSetOperators(ksp, rA.Get(), rA.Get()); CHKERRQ(ierr);
        if(ierr != 0)
            KRATOS_ERROR << "Error at KSPSetOperators, error code = " << ierr;

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
            std::cout << "KSPSetOperators completed" << std::endl;
        #endif

        /*
            Set linear solver defaults for this problem (optional).
            - By extracting the KSP and PC contexts from the KSP context,
            we can then directly call any KSP and PC routines to set
            various options.
            - The following two statements are optional; all of these
            parameters could alternatively be specified at runtime via
            KSPSetFromOptions().  All of these defaults can be
            overridden at runtime, as indicated below.
        */
        ierr = KSPSetTolerances(ksp, 1.0e-9, 1.0e-20, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
        if(ierr != 0)
            KRATOS_ERROR << "Error at KSPSetTolerances, error code = " << ierr;

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
            std::cout << "KSPSetTolerances completed" << std::endl;
        #endif

        /*
            Set PC
        */
        ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
        ierr = PCSetType(pc, PCFIELDSPLIT); CHKERRQ(ierr);
        ierr = ISCreateGeneral(Comm, mIndexU.size(), &mIndexU[0], PETSC_COPY_VALUES, &IS_u); CHKERRQ(ierr);
        ierr = ISCreateGeneral(Comm, mIndexWP.size(), &mIndexWP[0], PETSC_COPY_VALUES, &IS_wp); CHKERRQ(ierr);
        if(m_is_block)
        {
            ierr = ISSetBlockSize(IS_u, 3); CHKERRQ(ierr);
            ierr = ISSetBlockSize(IS_wp, 1); CHKERRQ(ierr);
            if(m_my_rank == 0)
                std::cout << "The block size is set for the sub-matrices" << std::endl;
        }
        ierr = PCFieldSplitSetIS(pc, "u", IS_u); CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc, "wp", IS_wp); CHKERRQ(ierr);

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
        {
            std::cout << "PCFIELDSPLIT completed" << std::endl;
        }
        std::cout << m_my_rank << ": mIndexU.size(): " << mIndexU.size() << std::endl;
        std::cout << m_my_rank << ": mIndexWP.size(): " << mIndexWP.size() << std::endl;
        #endif

        /*
            Set runtime options, e.g.,
                -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
            These options will override those specified above as long as
            KSPSetFromOptions() is called _after_ any other customization
            routines.
        */
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
            std::cout << "KSPSetFromOptions completed" << std::endl;
        #endif

        ierr = KSPSetUp(ksp); CHKERRQ(ierr);

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
            std::cout << "KSPSetUp completed" << std::endl;
        #endif

        #if defined(APPLY_NEAR_NULLSPACE) || defined(APPLY_COORDINATES)
        KSP*            ksp_all, ksp_U;
        PetscInt        nsplits; // should be 2
        Mat             A00;

        ierr = PCFieldSplitGetSubKSP(pc, &nsplits, &ksp_all); CHKERRQ(ierr);
//        KRATOS_WATCH(nsplits)
        ksp_U = ksp_all[0];
        #endif

        #ifdef APPLY_NEAR_NULLSPACE
        ierr = KSPGetOperators(ksp_U, &A00, PETSC_NULL); CHKERRQ(ierr);
        ierr = MatSetNearNullSpace(A00, mmatnull); CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&mmatnull); CHKERRQ(ierr);
//        mmatnull = PETSC_NULL;
        if(m_my_rank == 0)
            std::cout << "PetscFieldSplit_U_WP_Solver::" << __FUNCTION__ << ", the near nullspace for A00 is set" << std::endl;
        #endif

        #ifdef APPLY_COORDINATES
        PC              pc_U;
        ierr = KSPGetPC(ksp_U, &pc_U); CHKERRQ(ierr);
        PCSetCoordinates(pc_U, 3, static_cast<PetscInt>(mcoords.size()), &mcoords[0]);
        if(m_my_rank == 0)
            std::cout << "PetscFieldSplit_U_WP_Solver::" << __FUNCTION__ << ", PCSetCoordinates for A00 is set" << std::endl;
        #endif

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          Solve the linear system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        ierr = KSPSolve(ksp, rB.Get(), rX.Get());// CHKERRQ(ierr);
        if(ierr != 0)
            KRATOS_ERROR << "Error at KSPSolve, error code = " << ierr;

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
            std::cout << "KSPSolve completed" << std::endl;
        #endif

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          Check solution and clean up
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /*
            Check if the linear solver converged
        */
        KSPConvergedReason reason;
        ierr = KSPGetConvergedReason(ksp, &reason); // CHKERRQ(ierr);
        if(reason < 0)
        {
            KRATOS_ERROR << "The linear solver does not converge, reason = " << reason;
        }
        else
        {
            #ifdef DEBUG_SOLVER
            if(m_my_rank == 0)
                std::cout << "KSPSolve converged with reason = " << reason << std::endl;
            #endif
        }

        /*
            Check the error
        */
//        ierr = VecDuplicate(rB.Get(), &r); CHKERRQ(ierr);
//        ierr = MatMult(rA.Get(), rX.Get(), r); CHKERRQ(ierr);
//        ierr = VecAXPY(r, -1.0, rB.Get()); CHKERRQ(ierr);
//        ierr = VecNorm(rB.Get(), NORM_2, &norm_b); CHKERRQ(ierr);
//        ierr = VecNorm(r, NORM_2, &norm_r); CHKERRQ(ierr);
//        ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);

        /*
            Print convergence information.  PetscPrintf() produces a single
            print statement from all processes that share a communicator.
            An alternative is PetscFPrintf(), which prints to a file.
        */
//        ierr = PetscPrintf(Comm, "Norm of error %f iterations %d\n", norm_r/norm_b, its); CHKERRQ(ierr);

        /*
            Free work space.  All PETSc objects should be destroyed when they
            are no longer needed.
        */
        ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
//        ierr = VecDestroy(&r); CHKERRQ(ierr);

        return true;
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
        KRATOS_ERROR << "ERROR: This solver can be used for single RHS only";
        return false;
    }

    /**
     * Print information about this object.
     */
    void PrintInfo(std::ostream& rOStream) const override
    {
        if(m_my_rank == 0)
            rOStream << "PetscFieldSplit_U_WP solver finished.";
    }

    /**
     * Print object's data.
     */
    void  PrintData(std::ostream& rOStream) const override
    {
    }

private:

    int m_my_rank;
    bool m_is_block;
    std::vector<IndexType> mIndexU;
    std::vector<IndexType> mIndexWP;

    #ifdef APPLY_NEAR_NULLSPACE
    MatNullSpace    mmatnull;
    #endif

    #ifdef APPLY_COORDINATES
    std::vector<PetscReal> mcoords;
    #endif

    /**
     * Assignment operator.
     */
    PetscFieldSplit_U_WP_Solver& operator=(const PetscFieldSplit_U_WP_Solver& Other);
};

}  // namespace Kratos.

#undef DEBUG_SOLVER
#undef APPLY_NEAR_NULLSPACE
#undef APPLY_COORDINATES

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_WP_SOLVER_H_INCLUDED  defined
