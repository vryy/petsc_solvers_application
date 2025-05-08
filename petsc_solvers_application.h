//
//   Project Name:        Kratos
//   Last Modified by:    $Author: hbui $
//   Date:                $Date: Jul 16, 2013 $
//   Revision:            $Revision: 1.1 $
//
//
//Change log:
//Jul 16, 2013: change application name to petsc_solvers_application

#if !defined(KRATOS_PETSC_SOLVERS_APPLICATION_H_INCLUDED )
#define  KRATOS_PETSC_SOLVERS_APPLICATION_H_INCLUDED



// System includes
#include <string>
#include <iostream>


// External includes


// Project includes
#include "includes/define.h"
#include "includes/kratos_application.h"
#include "includes/variables.h"


namespace Kratos
{

    ///@name Kratos Globals
    ///@{

    // Variables definition

    ///@}
    ///@name Type Definitions
    ///@{

    ///@}
    ///@name  Enum's
    ///@{

    ///@}
    ///@name  Functions
    ///@{

    ///@}
    ///@name Kratos Classes
    ///@{

    /// Short class definition.
    /** Detail class definition.
    */
    class KratosPetscSolversApplication : public KratosApplication
    {
    public:
        ///@name Type Definitions
        ///@{

        KRATOS_CLASS_POINTER_DEFINITION(KratosPetscSolversApplication);

        ///@}
        ///@name Life Cycle
        ///@{

        /// Default constructor.
        KratosPetscSolversApplication();

        /// Destructor.
        ~KratosPetscSolversApplication() override
        {}


        ///@}
        ///@name Operators
        ///@{


        ///@}
        ///@name Operations
        ///@{

        void Register() override;

        ///@}
        ///@name Access
        ///@{


        ///@}
        ///@name Inquiry
        ///@{


        ///@}
        ///@name Input and output
        ///@{

        /// Turn back information as a string.
        std::string Info() const override
        {
            return "KratosPetscSolversApplication";
        }

        /// Print information about this object.
        void PrintInfo(std::ostream& rOStream) const override
        {
            rOStream << Info();
            PrintData(rOStream);
        }

        ///// Print object's data.
        void PrintData(std::ostream& rOStream) const override
        {
            KRATOS_WATCH("in KratosPetscSolversApplication");
            KRATOS_WATCH(KratosComponents<VariableData>::GetComponents().size() );
            rOStream << "Variables:" << std::endl;
            KratosComponents<VariableData>().PrintData(rOStream);
            rOStream << std::endl;
            rOStream << "Elements:" << std::endl;
            KratosComponents<Element>().PrintData(rOStream);
            rOStream << std::endl;
            rOStream << "Conditions:" << std::endl;
            KratosComponents<Condition>().PrintData(rOStream);
        }


        ///@}
        ///@name Friends
        ///@{


        ///@}

    protected:
        ///@name Protected static Member Variables
        ///@{


        ///@}
        ///@name Protected member Variables
        ///@{


        ///@}
        ///@name Protected Operators
        ///@{


        ///@}
        ///@name Protected Operations
        ///@{


        ///@}
        ///@name Protected  Access
        ///@{


        ///@}
        ///@name Protected Inquiry
        ///@{


        ///@}
        ///@name Protected LifeCycle
        ///@{


        ///@}

    private:
        ///@name Static Member Variables
        ///@{


        ///@}
        ///@name Member Variables
        ///@{


        ///@}
        ///@name Private Operators
        ///@{


        ///@}
        ///@name Private Operations
        ///@{


        ///@}
        ///@name Private  Access
        ///@{


        ///@}
        ///@name Private Inquiry
        ///@{


        ///@}
        ///@name Un accessible methods
        ///@{

        /// Assignment operator.
        KratosPetscSolversApplication& operator=(KratosPetscSolversApplication const& rOther);

        /// Copy constructor.
        KratosPetscSolversApplication(KratosPetscSolversApplication const& rOther);


        ///@}

    }; // Class KratosPetscSolversApplication

    ///@}


    ///@name Type Definitions
    ///@{


    ///@}
    ///@name Input and output
    ///@{

    ///@}


}  // namespace Kratos.

#endif // KRATOS_PETSC_SOLVERS_APPLICATION_H_INCLUDED  defined
