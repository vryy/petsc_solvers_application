/*
see petsc_solvers_application/LICENSE.txt
*/

//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: 5 Jan 2016 $
//   Revision:            $Revision: 1.1 $
//
//

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_GAMG_SOLVER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_GAMG_SOLVER_H_INCLUDED

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
#define USE_COORDS // this is for the geometric multigrid preconditioner

namespace Kratos
{

/**
This class constructs Petsc solver with GAMG preconditioner. Because the geometric information is used to build the multigrid hierarchy, this solver shall only be used with the parallel block builder and solver
This solver shall only be used with the elasticity problem
*/
template<class TSparseSpaceType, class TDenseSpaceType, class TModelPartType>
class PetscGAMGElasticitySolver : public LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscGAMGElasticitySolver);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TModelPartType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    typedef typename BaseType::DataType DataType;

    typedef typename BaseType::ModelPartType ModelPartType;

    /**
     * Default Constructor
     */
    PetscGAMGElasticitySolver()
    : BaseType(), m_my_rank(0), m_dim(3)
    {
    }

    /**
     * Destructor
     */
    ~PetscGAMGElasticitySolver() override
    {
    }

    // set the dimension for the geometric multigrid preconditioner
    void SetWorkingDimension(int Dim) {m_dim = Dim;}

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
//        m_nnodes = r_model_part.Nodes().size();
//
//        if(m_coords.size() != m_dim * m_nnodes)
//            m_coords.resize(m_dim * m_nnodes);
//
//        std::size_t cnt = 0;
//        if(m_dim == 2)
//        {
//            for(auto it = r_model_part.NodesBegin(); it != r_model_part.NodesEnd(); ++it)
//            {
//                m_coords[m_dim * cnt    ] = it->X0();
//                m_coords[m_dim * cnt + 1] = it->Y0();
//                ++cnt;
//            }
//            std::cout << "Coordinates is set for Petsc GAMG" << std::endl;
//        }
//        else if(m_dim == 3)
//        {
//            for(auto it = r_model_part.NodesBegin(); it != r_model_part.NodesEnd(); ++it)
//            {
//                m_coords[m_dim * cnt    ] = it->X0();
//                m_coords[m_dim * cnt + 1] = it->Y0();
//                m_coords[m_dim * cnt + 2] = it->Z0();
//                ++cnt;
//            }
//            std::cout << "Coordinates is set for Petsc GAMG" << std::endl;
//        }

        IndexType       Istart, Iend;
        MPI_Comm        Comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));
        PetscErrorCode  ierr;

        MPI_Comm_rank(Comm, &m_my_rank);

        ierr = MatGetOwnershipRange(rA.Get(), &Istart, &Iend); CHKERRV(ierr);
//        KRATOS_WATCH(Istart)
//        KRATOS_WATCH(Iend)
        if(m_coords.size() != (Iend - Istart))
            m_coords.resize(Iend - Istart);

        for(auto dof_iterator = rdof_set.begin(); dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            std::size_t node_id = dof_iterator->Id();
            std::size_t row_id = dof_iterator->EquationId();

            if((row_id >= Istart) && (row_id < Iend))
            {
                auto i_node = r_model_part.Nodes().find(node_id);
                if(i_node == r_model_part.Nodes().end())
                    KRATOS_ERROR << "The node does not exist in this partition. Probably data is consistent";

                double v = 0.0;
                if(dof_iterator->GetVariable() == DISPLACEMENT_X)
                    v = i_node->X0();
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Y)
                    v = i_node->Y0();
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Z)
                    v = i_node->Z0();
                m_coords[row_id - Istart] = v;
            }
        }
//        std::cout << "Generate coordinates completed" << std::endl;
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
        PetscReal       norm_b, norm_r;     /* ||b||, ||b-Ax|| */
        IndexType       its;
        PetscErrorCode  ierr;
        PetscScalar     v;
        MPI_Comm        Comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));

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

        ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
        ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr);
        ierr = PCSetCoordinates(pc, m_dim, m_nnodes, &m_coords[0]); CHKERRQ(ierr);
        KRATOS_WATCH(m_dim)
        KRATOS_WATCH(m_nnodes)
        KRATOS_WATCH(m_coords.size())
//        std::cout << "PCSetCoordinates is called" << std::endl;

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
            rOStream << "PetscGAMG solver finished.";
    }

    /**
     * Print object's data.
     */
    void PrintData(std::ostream& rOStream) const override
    {
    }

private:

    int m_my_rank;
    IndexType m_dim;
    IndexType m_nnodes;
    std::vector<PetscReal> m_coords;

    /**
     * Assignment operator.
     */
    PetscGAMGElasticitySolver& operator=(const PetscGAMGElasticitySolver& Other);
};

}  // namespace Kratos.

#undef DEBUG_SOLVER
#undef USE_COORDS

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_GAMG_SOLVER_H_INCLUDED  defined
