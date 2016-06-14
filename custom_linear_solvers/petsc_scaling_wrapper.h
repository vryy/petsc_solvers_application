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

template<class TSparseSpaceType, class TDenseSpaceType>
class PetscScalingWrapper : public LinearSolver<TSparseSpaceType, TDenseSpaceType>
{
public:

    KRATOS_CLASS_POINTER_DEFINITION(PetscScalingWrapper);

    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType> BaseType;

    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

    typedef typename TSparseSpaceType::VectorType VectorType;

    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

    typedef typename TSparseSpaceType::IndexType IndexType;

    typedef typename TSparseSpaceType::ValueType ValueType;

    /**
     * Default Constructor
     */
    PetscScalingWrapper(typename BaseType::Pointer pLinearSolver) : m_my_rank(0), mpLinearSolver(pLinearSolver)
    {
    }

    /**
     * Destructor
     */
    virtual ~PetscScalingWrapper()
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
    virtual bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB)
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
            rOStream << "Petsc scaling wrapper finished.";
    }

    /**
     * Print object's data.
     */
    virtual void PrintData(std::ostream& rOStream) const
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


/**
 * input stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType,class TReordererType>
inline std::istream& operator >> (std::istream& rIStream, PetscScalingWrapper<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    return rIStream;
}

/**
 * output stream function
 */
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream, const PetscScalingWrapper<TSparseSpaceType, TDenseSpaceType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}


}  // namespace Kratos.

#undef DEBUG_SOLVER

#endif // KRATOS_PETSC_SOLVERS_APP_PETSC_SCALING_WRAPPER_H_INCLUDED  defined 

