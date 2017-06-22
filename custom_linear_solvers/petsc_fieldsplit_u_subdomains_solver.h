/*
see petsc_solvers_application/LICENSE.txt
*/

//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: 6 Jun 2017 $
//   Revision:            $Revision: 1.0 $
//
//

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_SUBDOMAINS_SOLVER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_SUBDOMAINS_SOLVER_H_INCLUDED

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>

// External includes
#include "boost/smart_ptr.hpp"
#include <boost/foreach.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "petscksp.h"

// Project includes
#include "includes/define.h"
#include "includes/ublas_interface.h"
#include "utilities/openmp_utils.h"
#include "linear_solvers/linear_solver.h"

#define DEBUG_SOLVER
#define ENABLE_PROFILING

namespace Kratos
{

/**
This class constructs Petsc solver with PCFIELDSPLIT preconditioner.
This class assumes the provided DofSet is organized with contiguous tuple {ux,u_y,u_z}. Hence it shall only be used with the parallel block builder and solver.
*/
template<class TSparseSpaceType, class TDenseSpaceType>
class PetscFieldSplit_U_Subdomains_Solver : public LinearSolver<TSparseSpaceType, TDenseSpaceType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscFieldSplit_U_Subdomains_Solver);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    /**
     * Default Constructor
     */
    PetscFieldSplit_U_Subdomains_Solver(boost::python::list& subdomains_nodes,
            boost::python::list& subdomains_name,
            bool is_block)
    : m_my_rank(0), m_is_block(is_block)
    {
        ExtractSubdomainsNodes(subdomains_nodes, subdomains_name);
    }

    PetscFieldSplit_U_Subdomains_Solver(boost::python::list& subdomains_nodes,
            boost::python::list& subdomains_name)
    : m_my_rank(0), m_is_block(true)
    {
        ExtractSubdomainsNodes(subdomains_nodes, subdomains_name);
    }

    void ExtractSubdomainsNodes(boost::python::list& subdomains_nodes, boost::python::list& subdomains_name)
    {
        typedef boost::python::stl_input_iterator<boost::python::list> iterator1_value_type;
        BOOST_FOREACH(const iterator1_value_type::value_type& list_nodes,
                      std::make_pair(iterator1_value_type(subdomains_nodes), // begin
                      iterator1_value_type() ) ) // end
        {
            std::set<IndexType> subdomain_nodes;

            typedef boost::python::stl_input_iterator<int> iterator2_value_type;
            BOOST_FOREACH(const iterator2_value_type::value_type& node_id,
                          std::make_pair(iterator2_value_type(list_nodes), // begin
                          iterator2_value_type() ) ) // end
            {
                subdomain_nodes.insert(static_cast<IndexType>(node_id));
            }

            m_subdomains_nodes.push_back(subdomain_nodes);
        }

        typedef boost::python::stl_input_iterator<std::string> iterator3_value_type;
        BOOST_FOREACH(const iterator3_value_type::value_type& name,
                      std::make_pair(iterator3_value_type(subdomains_name), // begin
                      iterator3_value_type() ) ) // end
        {
            m_subdomains_name.push_back(name + "_u");
        }
    }

    /**
     * Destructor
     */
    virtual ~PetscFieldSplit_U_Subdomains_Solver()
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
        MPI_Comm        Comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));
        PetscErrorCode  ierr;

        MPI_Comm_rank(Comm, &m_my_rank);

        #ifdef ENABLE_PROFILING
        typename TSparseSpaceType::TimerType start = TSparseSpaceType::CreateTimer(TSparseSpaceType::GetComm(rA));
        #endif

        ierr = MatGetOwnershipRange(rA.Get(), &Istart, &Iend); CHKERRV(ierr);
//        KRATOS_WATCH(Istart)
//        KRATOS_WATCH(Iend)

//        std::cout << m_my_rank << ": " << " At PetscFieldSplit_U_Subdomains_Solver::ProvideAdditionalData" << std::endl;

        if(mIndexUSubdomains.size() != m_subdomains_nodes.size())
            mIndexUSubdomains.resize(m_subdomains_nodes.size());

        typedef std::map<std::size_t, std::vector<std::size_t> > rows_per_node_map_t;
        rows_per_node_map_t rows_per_node;
        std::set<std::size_t> remaining_nodes;
        for(typename ModelPart::DofsArrayType::iterator dof_iterator = rdof_set.begin();
                dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            std::size_t node_id = dof_iterator->Id();
            std::size_t row_id = dof_iterator->EquationId();

            if((row_id >= Istart) && (row_id < Iend))
            {
                if(    dof_iterator->GetVariable() == DISPLACEMENT_X
                    || dof_iterator->GetVariable() == DISPLACEMENT_Y
                    || dof_iterator->GetVariable() == DISPLACEMENT_Z )
                {
                    remaining_nodes.insert(node_id);
                    rows_per_node[node_id].push_back(row_id);
                }
            }
        }

        for(std::size_t i_subdomain = 0; i_subdomain < m_subdomains_nodes.size(); ++i_subdomain)
        {
            mIndexUSubdomains[i_subdomain].clear();
            for(typename std::set<IndexType>::iterator it = m_subdomains_nodes[i_subdomain].begin();
                    it != m_subdomains_nodes[i_subdomain].end(); ++it)
            {
                rows_per_node_map_t::iterator it2 = rows_per_node.find(*it);
                if(it2 != rows_per_node.end())
                {
                    for(std::size_t i = 0; i < it2->second.size(); ++i)
                        mIndexUSubdomains[i_subdomain].push_back(it2->second[i]);
                }
                remaining_nodes.erase(*it);
            }
        }

        std::vector<IndexType> IndexURemainingNodes;
        for(typename std::set<std::size_t>::iterator it = remaining_nodes.begin();
                it != remaining_nodes.end(); ++it)
        {
            rows_per_node_map_t::iterator it2 = rows_per_node.find(*it);
            if(it2 != rows_per_node.end())
            {
                for(std::size_t i = 0; i < it2->second.size(); ++i)
                    IndexURemainingNodes.push_back(it2->second[i]);
            }
        }
        mIndexUSubdomains.push_back(IndexURemainingNodes);
        m_subdomains_name.push_back("remaining_u");

        #ifdef ENABLE_PROFILING
        double elapsed_time = TSparseSpaceType::GetElapsedTime(start);
        start = TSparseSpaceType::CreateTimer(TSparseSpaceType::GetComm(rA));
//        if(m_my_rank == 0)
            std::cout << m_my_rank << ": PetscFieldSplit_U_Subdomains::ProvideAdditionalData indices extraction elapsed time: " << elapsed_time << std::endl;
        #endif

        // sort all the indices of all subdomains
        for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
        {
            std::sort(mIndexUSubdomains[i_subdomain].begin(), mIndexUSubdomains[i_subdomain].end());
        }

        #ifdef ENABLE_PROFILING
        elapsed_time = TSparseSpaceType::GetElapsedTime(start);
//        if(m_my_rank == 0)
            std::cout << m_my_rank << ": PetscFieldSplit_U_Subdomains::ProvideAdditionalData sorting elapsed time: " << elapsed_time << std::endl;
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
    virtual bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB)
    {
        Vec             r;             /* approx solution, RHS, A*x-b */
        KSP             ksp;               /* linear solver context */
        IS              IS_u_subdomain[mIndexUSubdomains.size()];  /* index set context */
//        IS              IS_u_1;
//        IS              IS_u_2;
        PC              pc;           /* preconditioner context */
        PetscReal       norm_b, norm_r;     /* ||b||, ||b-Ax|| */
        IndexType       its;
        IndexType       nsplits;
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

        #ifdef DEBUG_SOLVER
        for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
            std::cout << m_my_rank << ": mIndexUSubdomains[" << i_subdomain << "].size(): " << mIndexUSubdomains[i_subdomain].size() << std::endl;
//            PetscSynchronizedPrintf(Comm, "mIndexUSubdomains[%d].size(): %d\n", i_subdomain, mIndexUSubdomains[i_subdomain].size());
        #endif

        MPI_Barrier(Comm);

        for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
        {
            std::vector<IndexType>& IndexUSubdomain = mIndexUSubdomains[i_subdomain];
            ierr = ISCreateGeneral(Comm, IndexUSubdomain.size(), &IndexUSubdomain[0], PETSC_COPY_VALUES, &IS_u_subdomain[i_subdomain]); CHKERRQ(ierr);
        }

//        std::cout << m_my_rank << ": mIndexUSubdomains.size(): " << mIndexUSubdomains.size() << std::endl;
//        std::vector<IndexType>& IndexUSubdomain1 = mIndexUSubdomains[0];
//        std::vector<IndexType>& IndexUSubdomain2 = mIndexUSubdomains[1];
//        ierr = ISCreateGeneral(Comm, IndexUSubdomain1.size(), &IndexUSubdomain1[0], PETSC_COPY_VALUES, &IS_u_1); CHKERRQ(ierr);
//        ierr = ISCreateGeneral(Comm, IndexUSubdomain2.size(), &IndexUSubdomain2[0], PETSC_COPY_VALUES, &IS_u_2); CHKERRQ(ierr);

        MPI_Barrier(Comm);

        #ifdef DEBUG_SOLVER
//        if(m_my_rank == 0)
//            std::cout << "The IS for subdomains is created" << std::endl;
        std::cout << m_my_rank << ": The IS for subdomains is created" << std::endl;
        #endif

        if(m_is_block)
        {
            for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
            {
                ierr = ISSetBlockSize(IS_u_subdomain[i_subdomain], 3); CHKERRQ(ierr); // set block size for the subdomain block
            }

//            ierr = ISSetBlockSize(IS_u_1, 3); CHKERRQ(ierr);
//            ierr = ISSetBlockSize(IS_u_2, 3); CHKERRQ(ierr);
        }

        #ifdef DEBUG_SOLVER
//        if(m_my_rank == 0 && m_is_block)
//            std::cout << "The block size is set for the sub-matrices" << std::endl;
        if(m_is_block)
            std::cout << m_my_rank << ": The block size is set for the sub-matrices" << std::endl;
//        for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
//        {
//            std::stringstream ss;
//            PetscViewer viewer;
//            ss << "is_subdomain_" << i_subdomain << ".txt";
//            PetscViewerASCIIOpen(Comm, ss.str().c_str(), &viewer);
//            PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
//            ISView(IS_u_subdomain[i_subdomain], viewer);
//            PetscViewerDestroy(&viewer);
//            std::cout << m_my_rank << ": The view on subdomain " << i_subdomain << " is completed" << std::endl;
//        }
        #endif

        for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
        {
            ierr = PCFieldSplitSetIS(pc, m_subdomains_name[i_subdomain].c_str(), IS_u_subdomain[i_subdomain]); CHKERRQ(ierr);
        }

//        ierr = PCFieldSplitSetIS(pc, "building_u", IS_u_1); CHKERRQ(ierr);
//        ierr = PCFieldSplitSetIS(pc, "remaining_u", IS_u_2); CHKERRQ(ierr);

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
        {
            std::cout << m_my_rank << ": PCFIELDSPLIT completed" << std::endl;
        }
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
//        ierr = PetscPrintf(Comm, "Norm of error %f iterations %d\n", norm_r/norm_b, its); CHKERRQ(ierr);

        /*
            Free work space.  All PETSc objects should be destroyed when they
            are no longer needed.
        */

        for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
        {
            ierr = ISDestroy(&IS_u_subdomain[i_subdomain]); CHKERRQ(ierr);
        }
//        ierr = ISDestroy(&IS_u_1); CHKERRQ(ierr);
//        ierr = ISDestroy(&IS_u_2); CHKERRQ(ierr);
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
            rOStream << "PetscFieldSplit_U_Subdomains solver finished.";
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
    std::vector<std::set<IndexType> > m_subdomains_nodes;
    std::vector<std::string> m_subdomains_name;
    std::vector<std::vector<IndexType> > mIndexUSubdomains;

    /**
     * Assignment operator.
     */
    PetscFieldSplit_U_Subdomains_Solver& operator=(const PetscFieldSplit_U_Subdomains_Solver& Other);    
};


/**
 * input stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType,class TReordererType>
inline std::istream& operator >> (std::istream& rIStream, PetscFieldSplit_U_Subdomains_Solver<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    return rIStream;
}

/**
 * output stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream, const PetscFieldSplit_U_Subdomains_Solver<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}


}  // namespace Kratos.

#undef DEBUG_SOLVER

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_SUBDOMAINS_SOLVER_H_INCLUDED  defined 

