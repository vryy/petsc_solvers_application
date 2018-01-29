//   
//   Project Name:        Kratos       
//   Last Modified by:    $Author: hbui $
//   Date:                $Date:  $
//   Revision:            $Revision: 1.1 $
//
// 
//Change log:
//Jul 16, 2013: change application name to petsc_solvers_application


// System includes


// External includes 


// Project includes
#include "includes/define.h"
#include "includes/serializer.h"
#include "petsc_solvers_application/petsc_solvers_application.h"


namespace Kratos
{


    KratosPetscSolversApplication::KratosPetscSolversApplication()
    {}
    
    
 	void KratosPetscSolversApplication::Register()
 	{
 		// calling base class register to register Kratos components
 		KratosApplication::Register();
 		std::cout << "Initializing KratosPetscSolversApplication... " << std::endl;
        
        
        // TODO: add more constitutive law for Serializer
 	}

}  // namespace Kratos.


