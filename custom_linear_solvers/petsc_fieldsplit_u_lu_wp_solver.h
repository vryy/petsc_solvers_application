/*
see petsc_solvers_application/LICENSE.txt
*/

//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: 23 Jan 2017 $
//   Revision:            $Revision: 1.0 $
//
//

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_LU_WP_SOLVER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_LU_WP_SOLVER_H_INCLUDED

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

/**
This class constructs Petsc solver with PCFIELDSPLIT preconditioner.
This class assumes the provided DofSet is organized with contiguous tuple {ux,uy,uz}, {l_ux,l_uy,l_uz} and the water pressure is single field. Hence it shall only be used with the parallel block builder and solver.
*/
template<class TSparseSpaceType, class TDenseSpaceType, class TModelPartType>
class PetscFieldSplit_U_LU_WP_Solver : public LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscFieldSplit_U_LU_WP_Solver);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    typedef typename BaseType::ModelPartType ModelPartType;

    /**
     * Default Constructor
     */
    PetscFieldSplit_U_LU_WP_Solver()
    : BaseType(), m_my_rank(0), m_is_block(true)
    {
    }

    PetscFieldSplit_U_LU_WP_Solver(bool is_block)
    : BaseType(), m_my_rank(0), m_is_block(is_block)
    {
    }

    /**
     * Destructor
     */
    ~PetscFieldSplit_U_LU_WP_Solver() override
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
        mIndexLU.clear();
        mIndexWP.clear();
        for(auto dof_iterator = rdof_set.begin(); dof_iterator != rdof_set.end(); ++dof_iterator)
        {
//            std::size_t node_id = dof_iterator->Id();
            std::size_t row_id = dof_iterator->EquationId();

            if((row_id >= Istart) && (row_id < Iend))
            {
//                auto i_node = r_model_part.Nodes().find(node_id);
//                if(i_node == r_model_part.Nodes().end())
//                    KRATOS_ERROR << "The node does not exist in this partition. Probably data is consistent", "")

                if(dof_iterator->GetVariable() == DISPLACEMENT_X)
                {
                    mIndexU.push_back(row_id);
                }
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Y)
                {
                    mIndexU.push_back(row_id);
                }
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Z)
                {
                    mIndexU.push_back(row_id);
                }
                else if(dof_iterator->GetVariable() == LAGRANGE_DISPLACEMENT_X)
                {
                    mIndexLU.push_back(row_id);
                }
                else if(dof_iterator->GetVariable() == LAGRANGE_DISPLACEMENT_Y)
                {
                    mIndexLU.push_back(row_id);
                }
                else if(dof_iterator->GetVariable() == LAGRANGE_DISPLACEMENT_Z)
                {
                    mIndexLU.push_back(row_id);
                }
                else if(dof_iterator->GetVariable() == WATER_PRESSURE)
                {
                    mIndexWP.push_back(row_id);
                }
            }
        }

        std::sort(mIndexU.begin(), mIndexU.end());
        std::sort(mIndexLU.begin(), mIndexLU.end());
        std::sort(mIndexWP.begin(), mIndexWP.end());

        std::cout << m_my_rank << ": P: mIndexU.size(): " << mIndexU.size() << std::endl;
        std::cout << m_my_rank << ": P: mIndexLU.size(): " << mIndexLU.size() << std::endl;
        std::cout << m_my_rank << ": P: mIndexWP.size(): " << mIndexWP.size() << std::endl;
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
        PC              pc, pc_U_WP;           /* preconditioner context */
        IS              IS_u, IS_wp, IS_u_wp, IS_lu;   /* index set context */
        PetscReal       norm_b, norm_r;     /* ||b||, ||b-Ax|| */
        IndexType       its;
        PetscErrorCode  ierr;
        PetscScalar     v;
        MPI_Comm        Comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));
        std::vector<IndexType> IndexU_WP;
        KSP*            ksp_all, ksp_U_WP;
        PetscInt        dummy;

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

        /* The fieldsplit is organized as {{u, wp}, lu} */
        /*
            Set PC for outer fieldsplit
        */
        ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
        ierr = PCSetType(pc, PCFIELDSPLIT); CHKERRQ(ierr);
        IndexU_WP = mIndexU;
        IndexU_WP.insert(IndexU_WP.end(), mIndexWP.begin(), mIndexWP.end());
        std::sort(IndexU_WP.begin(), IndexU_WP.end());
        ierr = ISCreateGeneral(Comm, IndexU_WP.size(), &IndexU_WP[0], PETSC_COPY_VALUES, &IS_u_wp); CHKERRQ(ierr);
        ierr = ISCreateGeneral(Comm, mIndexLU.size(), &mIndexLU[0], PETSC_COPY_VALUES, &IS_lu); CHKERRQ(ierr);
        if(m_is_block)
        {
//            ierr = ISSetBlockSize(IS_u_wp, 4); CHKERRQ(ierr);
//            ierr = ISSetBlockSize(IS_lu, 3); CHKERRQ(ierr);
            if(m_my_rank == 0)
                std::cout << "The block size is not set for the sub-matrices LU" << std::endl;
        }
        ierr = PCFieldSplitSetIS(pc, "uwp", IS_u_wp); CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc, "lu", IS_lu); CHKERRQ(ierr);
        ierr = KSPSetUp(ksp); CHKERRQ(ierr);

        /*
            Set PC for inner fieldsplit
        */
        /* firstly rebuild the relative local index */
        std::vector<IndexType> LocalIndexU(mIndexU.size());
        std::vector<IndexType> LocalIndexWP(mIndexWP.size());
        std::map<IndexType, IndexType> index_map;
        for(std::size_t i = 0; i < IndexU_WP.size(); ++i)
            index_map[IndexU_WP[i]] = i;
        for(std::size_t i = 0; i < mIndexU.size(); ++i)
            LocalIndexU[i] = index_map[mIndexU[i]];
        for(std::size_t i = 0; i < mIndexWP.size(); ++i)
            LocalIndexWP[i] = index_map[mIndexWP[i]];

        /* compute the nested IS */
        ierr = PCFieldSplitGetSubKSP(pc, &dummy, &ksp_all); CHKERRQ(ierr);
        ksp_U_WP = ksp_all[0];
        ierr = ISCreateGeneral(Comm, LocalIndexU.size(), &LocalIndexU[0], PETSC_COPY_VALUES, &IS_u); CHKERRQ(ierr);
        ierr = ISCreateGeneral(Comm, LocalIndexWP.size(), &LocalIndexWP[0], PETSC_COPY_VALUES, &IS_wp); CHKERRQ(ierr);
        if(m_is_block)
        {
            ierr = ISSetBlockSize(IS_u, 3); CHKERRQ(ierr);
            ierr = ISSetBlockSize(IS_wp, 1); CHKERRQ(ierr);
            if(m_my_rank == 0)
                std::cout << "The block size is set for the sub-matrices U and WP" << std::endl;
        }
        ierr = KSPGetPC(ksp_U_WP, &pc_U_WP); CHKERRQ(ierr);
//        ierr = PCSetType(pc_U_WP, PCFIELDSPLIT); CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc_U_WP, "u", IS_u); CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc_U_WP, "wp", IS_wp); CHKERRQ(ierr);

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
        {
            std::cout << "PCFIELDSPLIT completed" << std::endl;
        }
//        std::cout << m_my_rank << ": mIndexU.size(): " << mIndexU.size() << std::endl;
//        std::cout << m_my_rank << ": mIndexLU.size(): " << mIndexLU.size() << std::endl;
//        std::cout << m_my_rank << ": mIndexWP.size(): " << mIndexWP.size() << std::endl;
        std::cout << m_my_rank << ": LocalIndexU.size(): " << LocalIndexU.size() << std::endl;
        std::cout << m_my_rank << ": LocalIndexWP.size(): " << LocalIndexWP.size() << std::endl;
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
            rOStream << "PetscFieldSplit_U_LU_WP solver finished.";
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
    std::vector<IndexType> mIndexLU;
    std::vector<IndexType> mIndexWP;

    /**
     * Assignment operator.
     */
    PetscFieldSplit_U_LU_WP_Solver& operator=(const PetscFieldSplit_U_LU_WP_Solver& Other);
};

}  // namespace Kratos.

#undef DEBUG_SOLVER

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_WP_SOLVER_H_INCLUDED  defined
