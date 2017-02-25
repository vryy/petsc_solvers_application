/*
==============================================================================
KratosStructuralApplication
A library based on:
Kratos
A General Purpose Software for Multi-Physics Finite Element Analysis
Version 1.0 (Released on march 05, 2007).

Copyright 2007
Pooyan Dadvand, Riccardo Rossi, Janosch Stascheit, Felix Nagel
pooyan@cimne.upc.edu
rrossi@cimne.upc.edu
- CIMNE (International Center for Numerical Methods in Engineering),
Gran Capita' s/n, 08034 Barcelona, Spain


Permission is hereby granted, free  of charge, to any person obtaining
a  copy  of this  software  and  associated  documentation files  (the
"Software"), to  deal in  the Software without  restriction, including
without limitation  the rights to  use, copy, modify,  merge, publish,
distribute,  sublicense and/or  sell copies  of the  Software,  and to
permit persons to whom the Software  is furnished to do so, subject to
the following condition:

Distribution of this code for  any  commercial purpose  is permissible
ONLY BY DIRECT ARRANGEMENT WITH THE COPYRIGHT OWNERS.

The  above  copyright  notice  and  this permission  notice  shall  be
included in all copies or substantial portions of the Software.

THE  SOFTWARE IS  PROVIDED  "AS  IS", WITHOUT  WARRANTY  OF ANY  KIND,
EXPRESS OR  IMPLIED, INCLUDING  BUT NOT LIMITED  TO THE  WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT  SHALL THE AUTHORS OR COPYRIGHT HOLDERS  BE LIABLE FOR ANY
CLAIM, DAMAGES OR  OTHER LIABILITY, WHETHER IN AN  ACTION OF CONTRACT,
TORT  OR OTHERWISE, ARISING  FROM, OUT  OF OR  IN CONNECTION  WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

==============================================================================
*/

//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: 14 Apr 2016 $
//   Revision:            $Revision: 1.0 $
//
//

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_LU_SOLVER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_LU_SOLVER_H_INCLUDED

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
#define VIEW_SUB_MATRICES

namespace Kratos
{

/**
This class constructs Petsc solver with PCFIELDSPLIT preconditioner.
This class assumes the provided DofSet is organized with contiguous tuple {ux,u_y,u_z} and the water pressure is single field. Hence it shall only be used with the parallel block builder and solver.
*/
template<class TSparseSpaceType, class TDenseSpaceType>
class PetscFieldSplit_U_LU_Solver : public LinearSolver<TSparseSpaceType, TDenseSpaceType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscFieldSplit_U_LU_Solver);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    /**
     * Default Constructor
     */
    PetscFieldSplit_U_LU_Solver() : m_my_rank(0), m_is_block(true)
    {
    }

    PetscFieldSplit_U_LU_Solver(bool is_block) : m_my_rank(0), m_is_block(is_block)
    {
    }

    /**
     * Destructor
     */
    virtual ~PetscFieldSplit_U_LU_Solver()
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
        // TODO collect the equation id for displacements and water pressure in the local process
        IndexType       Istart, Iend;
        MPI_Comm        comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));
        PetscErrorCode  ierr;

        ierr = MatGetOwnershipRange(rA.Get(), &Istart, &Iend); CHKERRV(ierr);
//        KRATOS_WATCH(Istart)
//        KRATOS_WATCH(Iend)

        mIndexU.clear();
        mIndexLU.clear();
        for(typename ModelPart::DofsArrayType::iterator dof_iterator = rdof_set.begin();
                dof_iterator != rdof_set.end(); ++dof_iterator)
        {
//            std::size_t node_id = dof_iterator->Id();
            std::size_t row_id = dof_iterator->EquationId();

            if((row_id >= Istart) && (row_id < Iend))
            {
//                ModelPart::NodesContainerType::iterator i_node = r_model_part.Nodes().find(node_id);
//                if(i_node == r_model_part.Nodes().end())
//                    KRATOS_THROW_ERROR(std::logic_error, "The node does not exist in this partition. Probably data is consistent", "")

                if(dof_iterator->GetVariable() == DISPLACEMENT_X)
                    mIndexU.push_back(row_id);
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Y)
                    mIndexU.push_back(row_id);
                else if(dof_iterator->GetVariable() == DISPLACEMENT_Z)
                    mIndexU.push_back(row_id);
                else if(dof_iterator->GetVariable() == LAGRANGE_DISPLACEMENT_X)
                    mIndexLU.push_back(row_id);
                else if(dof_iterator->GetVariable() == LAGRANGE_DISPLACEMENT_Y)
                    mIndexLU.push_back(row_id);
                else if(dof_iterator->GetVariable() == LAGRANGE_DISPLACEMENT_Z)
                    mIndexLU.push_back(row_id);
            }
        }
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
        Vec             r;             /* approx solution, RHS, A*x-b */
        KSP             ksp;               /* linear solver context */
        PC              pc;           /* preconditioner context */
        IS              IS_u, IS_lu;   /* index set context */
        PetscReal       norm_b, norm_r;     /* ||b||, ||b-Ax|| */
        IndexType       its;
        PetscErrorCode  ierr;
        PetscScalar     v;
        MPI_Comm        comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));
        PetscViewer     viewer;
        #ifdef VIEW_SUB_MATRICES
        Mat             A00, A01, A10, A11;
        #endif

        MPI_Comm_rank(comm, &m_my_rank);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* 
            Create linear solver context
        */
        ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
        if(ierr != 0)
            KRATOS_THROW_ERROR(std::runtime_error, "Error at KSPCreate, error code =", ierr)

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
            KRATOS_THROW_ERROR(std::runtime_error, "Error at KSPSetOperators, error code =", ierr)

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
            KRATOS_THROW_ERROR(std::runtime_error, "Error at KSPSetTolerances, error code =", ierr)

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
            std::cout << "KSPSetTolerances completed" << std::endl;
        #endif

        /* 
            Set PC
        */
        ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
        ierr = PCSetType(pc, PCFIELDSPLIT); CHKERRQ(ierr);
        ierr = ISCreateGeneral(comm, mIndexU.size(), &mIndexU[0], PETSC_COPY_VALUES, &IS_u); CHKERRQ(ierr);
        ierr = ISCreateGeneral(comm, mIndexLU.size(), &mIndexLU[0], PETSC_COPY_VALUES, &IS_lu); CHKERRQ(ierr);
        if(m_is_block)
        {
            ierr = ISSetBlockSize(IS_u, 3); CHKERRQ(ierr); // set block size for the u block
            ierr = ISSetBlockSize(IS_lu, 3); CHKERRQ(ierr); // set block size for the lagrange-u block
            if(m_my_rank == 0)
                std::cout << "The block size is set for the sub-matrices" << std::endl;
        }
        ierr = PCFieldSplitSetIS(pc, "u", IS_u); CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc, "lu", IS_lu); CHKERRQ(ierr);

        #ifdef VIEW_SUB_MATRICES
        ierr = MatGetSubMatrix(rA.Get(), IS_u, IS_u,   MAT_INITIAL_MATRIX, &A00); CHKERRQ(ierr);
        ierr = MatGetSubMatrix(rA.Get(), IS_u, IS_lu,  MAT_INITIAL_MATRIX, &A01); CHKERRQ(ierr);
        ierr = MatGetSubMatrix(rA.Get(), IS_lu, IS_u,  MAT_INITIAL_MATRIX, &A10); CHKERRQ(ierr);
        ierr = MatGetSubMatrix(rA.Get(), IS_lu, IS_lu, MAT_INITIAL_MATRIX, &A11); CHKERRQ(ierr);

        ierr = PetscViewerASCIIOpen(comm, "A00.m", &viewer); CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
        ierr = MatView(A00, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

        ierr = PetscViewerASCIIOpen(comm, "A01.m", &viewer); CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
        ierr = MatView(A01, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

        ierr = PetscViewerASCIIOpen(comm, "A10.m", &viewer); CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
        ierr = MatView(A10, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

        ierr = PetscViewerASCIIOpen(comm, "A11.m", &viewer); CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
        ierr = MatView(A11, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        #endif

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
        {
            std::cout << "PCFIELDSPLIT completed" << std::endl;
        }
        std::cout << m_my_rank << ": mIndexU.size(): " << mIndexU.size() << std::endl;
        std::cout << m_my_rank << ": mIndexLU.size(): " << mIndexLU.size() << std::endl;
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

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                          Solve the linear system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        ierr = KSPSolve(ksp, rB.Get(), rX.Get());// CHKERRQ(ierr);
        if(ierr != 0)
            KRATOS_THROW_ERROR(std::runtime_error, "Error at KSPSolve, error code =", ierr)

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
            KRATOS_THROW_ERROR(std::runtime_error, "The linear solver does not converge, reason =", reason)
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
//        ierr = PetscPrintf(comm, "Norm of error %f iterations %d\n", norm_r/norm_b, its); CHKERRQ(ierr);

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
    virtual bool Solve(SparseMatrixType& rA, DenseMatrixType& rX, DenseMatrixType& rB)
    {
        KRATOS_THROW_ERROR(std::logic_error, "ERROR: This solver can be used for single RHS only", "");
        return false;
    }

    /**
     * Print information about this object.
     */
    virtual void PrintInfo(std::ostream& rOStream) const
    {
        if(m_my_rank == 0)
            rOStream << "PetscFieldSplit_U_LU solver finished.";
    }

    /**
     * Print object's data.
     */
    virtual void  PrintData(std::ostream& rOStream) const
    {
    }

private:

    int m_my_rank;
    bool m_is_block;
    std::vector<IndexType> mIndexU;
    std::vector<IndexType> mIndexLU;

    /**
     * Assignment operator.
     */
    PetscFieldSplit_U_LU_Solver& operator=(const PetscFieldSplit_U_LU_Solver& Other);    
};


/**
 * input stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType,class TReordererType>
inline std::istream& operator >> (std::istream& rIStream, PetscFieldSplit_U_LU_Solver<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    return rIStream;
}

/**
 * output stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream, const PetscFieldSplit_U_LU_Solver<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}


}  // namespace Kratos.

#undef VIEW_SUB_MATRICES
#undef DEBUG_SOLVER

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_LU_SOLVER_H_INCLUDED  defined 
