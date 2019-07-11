#include <iostream>
#include <string>
#include <sstream>

#ifndef CARLICENSEPLATEDETECTION_LPREGISTRAR_H
#define CARLICENSEPLATEDETECTION_LPREGISTRAR_H

using namespace std;

namespace Registrar {

    class LPRegistrar {

    public :
        bool registerLP(string d2, string letter, string d3, string region) {
            stringstream LP;
            LP << d2 << "-" << letter << "-" << d3 << "-" << region;
            cout << "Registered LP: " << LP.str() << endl ;
            return true;
        }

    };
}

#endif //CARLICENSEPLATEDETECTION_LPREGISTRAR_H
