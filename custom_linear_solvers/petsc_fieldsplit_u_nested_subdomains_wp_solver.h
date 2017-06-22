/*
see petsc_solvers_application/LICENSE.txt
*/

//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: 5 Jun 2016 $
//   Revision:            $Revision: 1.0 $
//
//

#if !defined(KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_NESTED_SUBDOMAINS_WP_SOLVER_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_NESTED_SUBDOMAINS_WP_SOLVER_H_INCLUDED

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
This class assumes the provided DofSet is organized with contiguous tuple {ux,u_y,u_z} and the water pressure is single field. Hence it shall only be used with the parallel block builder and solver.
*/
template<class TSparseSpaceType, class TDenseSpaceType>
class PetscFieldSplit_U_Nested_Subdomains_WP_Solver : public LinearSolver<TSparseSpaceType, TDenseSpaceType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscFieldSplit_U_Nested_Subdomains_WP_Solver);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    /**
     * Default Constructor
     */
    PetscFieldSplit_U_Nested_Subdomains_WP_Solver(boost::python::list& subdomains_nodes,
            boost::python::list& subdomains_name,
            bool is_block)
    : m_my_rank(0), m_is_block(is_block)
    {
        ExtractSubdomainsNodes(subdomains_nodes, subdomains_name);
    }

    PetscFieldSplit_U_Nested_Subdomains_WP_Solver(boost::python::list& subdomains_nodes,
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
            m_subdomains_name.push_back(name);
        }
    }

    /**
     * Destructor
     */
    virtual ~PetscFieldSplit_U_Nested_Subdomains_WP_Solver()
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

        int mpi_size;
        MPI_Comm_size(Comm, &mpi_size);

        #ifdef ENABLE_PROFILING
        typename TSparseSpaceType::TimerType start = TSparseSpaceType::CreateTimer(TSparseSpaceType::GetComm(rA));
        #endif

        ierr = MatGetOwnershipRange(rA.Get(), &Istart, &Iend); CHKERRV(ierr);
//        KRATOS_WATCH(Istart)
//        KRATOS_WATCH(Iend)

        mIndexU.clear();
        mIndexWP.clear();

        // collect all the row_id of the displacement and water pressure
        std::set<IndexType> IndexU;
        for(typename ModelPart::DofsArrayType::iterator dof_iterator = rdof_set.begin();
                dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            std::size_t node_id = dof_iterator->Id();
            std::size_t row_id = dof_iterator->EquationId();

            if((row_id >= Istart) && (row_id < Iend))
            {
                if(    dof_iterator->GetVariable() == DISPLACEMENT_X
                    || dof_iterator->GetVariable() == DISPLACEMENT_Y
                    || dof_iterator->GetVariable() == DISPLACEMENT_Z  )
                {
                    IndexU.insert(row_id);
                }
                else if(dof_iterator->GetVariable() == WATER_PRESSURE)
                {
                    mIndexWP.push_back(row_id);
                }
            }
        }

        if(mIndexU.size() != IndexU.size())
            mIndexU.resize(IndexU.size());
        std::copy(IndexU.begin(), IndexU.end(), mIndexU.begin());

        // commmunicate and find the first row of the subdomain block
        std::cout << m_my_rank << ": mIndexU.size(): " << mIndexU.size() << std::endl;
        std::vector<IndexType> all_u_sizes(mpi_size);
        IndexType u_size = static_cast<IndexType>(mIndexU.size());
        MPI_Allgather(&u_size, 1, MPI_INT, &all_u_sizes[0], 1, MPI_INT, Comm);
        std::cout << m_my_rank << ": all_u_sizes:";
        for(std::size_t i_rank = 0; i_rank < mpi_size; ++i_rank)
            std::cout << " " << all_u_sizes[i_rank];
        std::cout << std::endl;

        IndexType first_row_subdomain = 0;
        for(int i_rank = 0; i_rank < m_my_rank; ++i_rank)
            first_row_subdomain += all_u_sizes[i_rank];

        // create the local map from global row_id to the local row_id amongst the subdomains
        std::map<IndexType, IndexType> map_row_id;
        for(typename std::set<IndexType>::iterator it = IndexU.begin(); it != IndexU.end(); ++it)
            map_row_id[*it] = first_row_subdomain++;

        // assign the index for each subdomain
        if(mIndexUSubdomains.size() != m_subdomains_nodes.size())
            mIndexUSubdomains.resize(m_subdomains_nodes.size());
        for(std::size_t i_subdomain = 0; i_subdomain < m_subdomains_nodes.size(); ++i_subdomain)
        {
            mIndexUSubdomains[i_subdomain].clear();
        }

        std::set<IndexType> IndexUNonSubdomain;

        for(typename ModelPart::DofsArrayType::iterator dof_iterator = rdof_set.begin();
                dof_iterator != rdof_set.end(); ++dof_iterator)
        {
            std::size_t node_id = dof_iterator->Id();
            std::size_t row_id = dof_iterator->EquationId();

            if((row_id >= Istart) && (row_id < Iend))
            {
//                ModelPart::NodesContainerType::iterator i_node = r_model_part.Nodes().find(node_id);
//                if(i_node == r_model_part.Nodes().end())
//                    KRATOS_THROW_ERROR(std::logic_error, "The node does not exist in this partition. Probably data is consistent", "")

                if(    dof_iterator->GetVariable() == DISPLACEMENT_X
                    || dof_iterator->GetVariable() == DISPLACEMENT_Y
                    || dof_iterator->GetVariable() == DISPLACEMENT_Z  )
                {
                    bool found_node = false;
                    for(std::size_t i_subdomain = 0; i_subdomain < m_subdomains_nodes.size(); ++i_subdomain)
                    {
                        if(std::find(m_subdomains_nodes[i_subdomain].begin(),
                                m_subdomains_nodes[i_subdomain].end(), node_id) != m_subdomains_nodes[i_subdomain].end())
                        {
                            mIndexUSubdomains[i_subdomain].push_back(map_row_id[row_id]);
                            found_node = true;
                            break;
                        }
                    }

                    if(found_node == false)
                    {
//                        std::stringstream ss;
//                        ss << "Node " << node_id << " at row " << row_id << " is not found in all the subdomains";
//                        KRATOS_THROW_ERROR(std::logic_error, ss.str(), "")

//                        IndexUNonSubdomain.insert(row_id);
                        IndexUNonSubdomain.insert(map_row_id[row_id]);
                    }
                }
            }
        }

//        std::cout << m_my_rank << ": " << " extract dofs completed" << std::endl;

        if(IndexUNonSubdomain.size() != 0)
        {
            std::vector<IndexType> IndexUNonSubdomainVect(IndexUNonSubdomain.begin(), IndexUNonSubdomain.end());
            mIndexUSubdomains.push_back(IndexUNonSubdomainVect);
            m_subdomains_name.push_back("remaining");
        }
        std::cout << m_my_rank << ": " << "IndexUNonSubdomain.size(): " << IndexUNonSubdomain.size() << std::endl;

        // make a quick check
//        if(row_local_subdomain != mIndexU.size())
//        {
//            std::stringstream ss;
//            ss << "(row_local_subdomain = " << row_local_subdomain << ") != (mIndexU.size() = " << mIndexU.size() << ")"
//               << ". Something must be wrong";
//            KRATOS_THROW_ERROR(std::logic_error, ss.str(), "")
//        }
    }

//    virtual void ProvideAdditionalData(
//        SparseMatrixType& rA,
//        VectorType& rX,
//        VectorType& rB,
//        typename ModelPart::DofsArrayType& rdof_set,
//        ModelPart& r_model_part
//    )
//    {
//        // TODO collect the equation id for displacements and water pressure in the local process
//        IndexType       Istart, Iend;
//        MPI_Comm        Comm = TSparseSpaceType::ExtractComm(TSparseSpaceType::GetComm(rA));
//        PetscErrorCode  ierr;

//        MPI_Comm_rank(Comm, &m_my_rank);

//        #ifdef ENABLE_PROFILING
//        typename TSparseSpaceType::TimerType start = TSparseSpaceType::CreateTimer(TSparseSpaceType::GetComm(rA));
//        #endif

//        ierr = MatGetOwnershipRange(rA.Get(), &Istart, &Iend); CHKERRV(ierr);
////        KRATOS_WATCH(Istart)
////        KRATOS_WATCH(Iend)

//        mIndexU.clear();
//        if(mIndexUSubdomains.size() != m_subdomains_nodes.size())
//            mIndexUSubdomains.resize(m_subdomains_nodes.size());
//        for(std::size_t i_subdomain = 0; i_subdomain < m_subdomains_nodes.size(); ++i_subdomain)
//        {
//            mIndexUSubdomains[i_subdomain].clear();
//        }
//        mIndexWP.clear();

//        std::set<IndexType> IndexUNonSubdomain;

//        IndexType row_local_subdomain = 0;

//        for(typename ModelPart::DofsArrayType::iterator dof_iterator = rdof_set.begin();
//                dof_iterator != rdof_set.end(); ++dof_iterator)
//        {
//            std::size_t node_id = dof_iterator->Id();
//            std::size_t row_id = dof_iterator->EquationId();

//            if((row_id >= Istart) && (row_id < Iend))
//            {
////                ModelPart::NodesContainerType::iterator i_node = r_model_part.Nodes().find(node_id);
////                if(i_node == r_model_part.Nodes().end())
////                    KRATOS_THROW_ERROR(std::logic_error, "The node does not exist in this partition. Probably data is consistent", "")

//                if(    dof_iterator->GetVariable() == DISPLACEMENT_X
//                    || dof_iterator->GetVariable() == DISPLACEMENT_Y
//                    || dof_iterator->GetVariable() == DISPLACEMENT_Z  )
//                {
//                    mIndexU.push_back(row_id);

//                    bool found_node = false;
//                    for(std::size_t i_subdomain = 0; i_subdomain < m_subdomains_nodes.size(); ++i_subdomain)
//                    {
//                        if(std::find(m_subdomains_nodes[i_subdomain].begin(),
//                                m_subdomains_nodes[i_subdomain].end(), node_id) != m_subdomains_nodes[i_subdomain].end())
//                        {
////                            mIndexUSubdomains[i_subdomain].push_back(row_id);
//                            mIndexUSubdomains[i_subdomain].push_back(row_local_subdomain++);
//                            found_node = true;
//                            break;
//                        }
//                    }

//                    if(found_node == false)
//                    {
////                        std::stringstream ss;
////                        ss << "Node " << node_id << " at row " << row_id << " is not found in all the subdomains";
////                        KRATOS_THROW_ERROR(std::logic_error, ss.str(), "")

////                        IndexUNonSubdomain.insert(row_id);
//                        IndexUNonSubdomain.insert(row_local_subdomain++);
//                    }
//                }
//                else if(dof_iterator->GetVariable() == WATER_PRESSURE)
//                {
//                    mIndexWP.push_back(row_id);
//                }
//            }
//        }

////        std::cout << m_my_rank << ": " << " extract dofs completed" << std::endl;

//        if(IndexUNonSubdomain.size() != 0)
//        {
//            std::vector<IndexType> IndexUNonSubdomainVect(IndexUNonSubdomain.begin(), IndexUNonSubdomain.end());
//            mIndexUSubdomains.push_back(IndexUNonSubdomainVect);
//            m_subdomains_name.push_back("remaining");
//        }
//        std::cout << m_my_rank << ": " << "IndexUNonSubdomain.size(): " << IndexUNonSubdomain.size() << std::endl;

//        // make a quick check
//        if(row_local_subdomain != mIndexU.size())
//        {
//            std::stringstream ss;
//            ss << "(row_local_subdomain = " << row_local_subdomain << ") != (mIndexU.size() = " << mIndexU.size() << ")"
//               << ". Something must be wrong";
//            KRATOS_THROW_ERROR(std::logic_error, ss.str(), "")
//        }
//    }

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
        KSP             *subksp;      /* sub KSPs */
        PC              pc, pcU;           /* preconditioner context */
        IS              IS_u, IS_wp;   /* index set context */
        PetscReal       norm_b, norm_r;     /* ||b||, ||b-Ax|| */
        IndexType       its, nsplits;
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
//        ierr = PCSetUp(pc); CHKERRQ(ierr);

        // set fieldsplit pc for u block
        ierr = PCFieldSplitGetSubKSP(pc, &nsplits, &subksp); CHKERRQ(ierr);
        ierr = KSPGetPC(subksp[0], &pcU); CHKERRQ(ierr);
        ierr = PCSetType(pcU, PCFIELDSPLIT); CHKERRQ(ierr);
        for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
        {
            IS IS_u_subdomain;

            ierr = ISCreateGeneral(Comm, mIndexUSubdomains[i_subdomain].size(), &mIndexUSubdomains[i_subdomain][0], PETSC_COPY_VALUES, &IS_u_subdomain); CHKERRQ(ierr);

            if(m_is_block)
            {
                ierr = ISSetBlockSize(IS_u_subdomain, 3); CHKERRQ(ierr); // set block size for the subdomain block
            }

            ierr = PCFieldSplitSetIS(pcU, m_subdomains_name[i_subdomain].c_str(), IS_u_subdomain); CHKERRQ(ierr);

            ierr = ISDestroy(&IS_u_subdomain); CHKERRQ(ierr);
        }

//        ierr = PCSetUp(pcU); CHKERRQ(ierr);

        ierr = PetscFree(subksp); CHKERRQ(ierr);
        ierr = ISDestroy(&IS_u); CHKERRQ(ierr);
        ierr = ISDestroy(&IS_wp); CHKERRQ(ierr);

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0 && m_is_block)
            std::cout << "The block size is set for the sub-matrices" << std::endl;
        #endif

        #ifdef DEBUG_SOLVER
        if(m_my_rank == 0)
        {
            std::cout << "PCFIELDSPLIT completed" << std::endl;
        }
        std::cout << m_my_rank << ": mIndexU.size(): " << mIndexU.size() << std::endl;
        for(std::size_t i_subdomain = 0; i_subdomain < mIndexUSubdomains.size(); ++i_subdomain)
            std::cout << m_my_rank << ": mIndexUSubdomains[" << i_subdomain << "].size(): " << mIndexUSubdomains[i_subdomain].size() << std::endl;
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
            rOStream << "PetscFieldSplit_U_Nested_Subdomains_WP solver finished.";
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
    std::vector<IndexType> mIndexU; // = union of mIndexUSubdomains
    std::vector<std::vector<IndexType> > mIndexUSubdomains;
    std::vector<IndexType> mIndexWP;

    /**
     * Assignment operator.
     */
    PetscFieldSplit_U_Nested_Subdomains_WP_Solver& operator=(const PetscFieldSplit_U_Nested_Subdomains_WP_Solver& Other);    
};


/**
 * input stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType,class TReordererType>
inline std::istream& operator >> (std::istream& rIStream, PetscFieldSplit_U_Nested_Subdomains_WP_Solver<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    return rIStream;
}

/**
 * output stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream, const PetscFieldSplit_U_Nested_Subdomains_WP_Solver<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}


}  // namespace Kratos.

#undef DEBUG_SOLVER

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_FIELDSPLIT_U_NESTED_SUBDOMAINS_WP_SOLVER_H_INCLUDED  defined 

