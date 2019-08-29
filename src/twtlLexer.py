# $ANTLR 3.1.3 Mar 18, 2009 10:09:25 C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g 2016-02-22 17:58:48

import sys
from antlr3 import *
from antlr3.compat import set, frozenset
                         
license_text='''
    Lexer for TWTL formulae. 
    Copyright (C) 2015-2016  Cristian Ioan Vasile <cvasile@bu.edu>
    Hybrid and Networked Systems (HyNeSs) Group, BU Robotics Lab,
    Boston University

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''



# for convenience in actions
HIDDEN = BaseRecognizer.HIDDEN

# token types
LETTER=18
CONCAT=9
PROP=11
WITHIN=8
LWLETTER=15
INT=10
WS=20
EOF=-1
HOLD=7
LINECMT=19
OR=5
TRUE=12
DIGIT=14
NOT=6
T__26=26
HGLETTER=16
AND=4
T__22=22
T__23=23
T__24=24
FALSE=13
T__25=25
HGLETTERALL=17
T__21=21


class twtlLexer(Lexer):

    grammarFileName = "C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g"
    antlr_version = version_str_to_tuple("3.1.3 Mar 18, 2009 10:09:25")
    antlr_version_str = "3.1.3 Mar 18, 2009 10:09:25"

    def __init__(self, input=None, state=None):
        if state is None:
            state = RecognizerSharedState()
        super(twtlLexer, self).__init__(input, state)


        self.dfa11 = self.DFA11(
            self, 11,
            eot = self.DFA11_eot,
            eof = self.DFA11_eof,
            min = self.DFA11_min,
            max = self.DFA11_max,
            accept = self.DFA11_accept,
            special = self.DFA11_special,
            transition = self.DFA11_transition
            )




                               
    def getAlphabet(self):
        return self.alphabet

    def setAlphabet(self, alphabet):
        self.alphabet = alphabet



    # $ANTLR start "AND"
    def mAND(self, ):

        try:
            _type = AND
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:36:5: ( '&' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:36:7: '&'
            pass 
            self.match(38)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "AND"



    # $ANTLR start "OR"
    def mOR(self, ):

        try:
            _type = OR
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:37:4: ( '|' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:37:6: '|'
            pass 
            self.match(124)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "OR"



    # $ANTLR start "NOT"
    def mNOT(self, ):

        try:
            _type = NOT
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:38:5: ( '!' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:38:7: '!'
            pass 
            self.match(33)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "NOT"



    # $ANTLR start "HOLD"
    def mHOLD(self, ):

        try:
            _type = HOLD
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:39:6: ( 'H' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:39:8: 'H'
            pass 
            self.match(72)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "HOLD"



    # $ANTLR start "WITHIN"
    def mWITHIN(self, ):

        try:
            _type = WITHIN
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:40:8: ( 'W' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:40:10: 'W'
            pass 
            self.match(87)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "WITHIN"



    # $ANTLR start "CONCAT"
    def mCONCAT(self, ):

        try:
            _type = CONCAT
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:41:8: ( '*' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:41:10: '*'
            pass 
            self.match(42)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "CONCAT"



    # $ANTLR start "T__21"
    def mT__21(self, ):

        try:
            _type = T__21
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:42:7: ( '^' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:42:9: '^'
            pass 
            self.match(94)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "T__21"



    # $ANTLR start "T__22"
    def mT__22(self, ):

        try:
            _type = T__22
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:43:7: ( '[' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:43:9: '['
            pass 
            self.match(91)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "T__22"



    # $ANTLR start "T__23"
    def mT__23(self, ):

        try:
            _type = T__23
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:44:7: ( ']' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:44:9: ']'
            pass 
            self.match(93)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "T__23"



    # $ANTLR start "T__24"
    def mT__24(self, ):

        try:
            _type = T__24
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:45:7: ( ',' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:45:9: ','
            pass 
            self.match(44)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "T__24"



    # $ANTLR start "T__25"
    def mT__25(self, ):

        try:
            _type = T__25
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:46:7: ( '(' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:46:9: '('
            pass 
            self.match(40)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "T__25"



    # $ANTLR start "T__26"
    def mT__26(self, ):

        try:
            _type = T__26
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:47:7: ( ')' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:47:9: ')'
            pass 
            self.match(41)



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "T__26"



    # $ANTLR start "DIGIT"
    def mDIGIT(self, ):

        try:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:132:13: ( ( '0' .. '9' ) )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:132:15: ( '0' .. '9' )
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:132:15: ( '0' .. '9' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:132:16: '0' .. '9'
            pass 
            self.matchRange(48, 57)







        finally:

            pass

    # $ANTLR end "DIGIT"



    # $ANTLR start "LWLETTER"
    def mLWLETTER(self, ):

        try:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:134:13: ( ( 'a' .. 'z' ) )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:134:15: ( 'a' .. 'z' )
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:134:15: ( 'a' .. 'z' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:134:16: 'a' .. 'z'
            pass 
            self.matchRange(97, 122)







        finally:

            pass

    # $ANTLR end "LWLETTER"



    # $ANTLR start "HGLETTER"
    def mHGLETTER(self, ):

        try:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:136:13: ( ( 'A' .. 'Z' ) )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:136:15: ( 'A' .. 'Z' )
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:136:15: ( 'A' .. 'Z' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:136:16: 'A' .. 'Z'
            pass 
            self.matchRange(65, 90)







        finally:

            pass

    # $ANTLR end "HGLETTER"



    # $ANTLR start "HGLETTERALL"
    def mHGLETTERALL(self, ):

        try:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:13: ( ( 'A' .. 'G' ) | ( 'I' .. 'V' ) | ( 'X' .. 'Z' ) )
            alt1 = 3
            LA1 = self.input.LA(1)
            if LA1 == 65 or LA1 == 66 or LA1 == 67 or LA1 == 68 or LA1 == 69 or LA1 == 70 or LA1 == 71:
                alt1 = 1
            elif LA1 == 73 or LA1 == 74 or LA1 == 75 or LA1 == 76 or LA1 == 77 or LA1 == 78 or LA1 == 79 or LA1 == 80 or LA1 == 81 or LA1 == 82 or LA1 == 83 or LA1 == 84 or LA1 == 85 or LA1 == 86:
                alt1 = 2
            elif LA1 == 88 or LA1 == 89 or LA1 == 90:
                alt1 = 3
            else:
                nvae = NoViableAltException("", 1, 0, self.input)

                raise nvae

            if alt1 == 1:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:15: ( 'A' .. 'G' )
                pass 
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:15: ( 'A' .. 'G' )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:16: 'A' .. 'G'
                pass 
                self.matchRange(65, 71)





            elif alt1 == 2:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:28: ( 'I' .. 'V' )
                pass 
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:28: ( 'I' .. 'V' )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:29: 'I' .. 'V'
                pass 
                self.matchRange(73, 86)





            elif alt1 == 3:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:43: ( 'X' .. 'Z' )
                pass 
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:43: ( 'X' .. 'Z' )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:138:44: 'X' .. 'Z'
                pass 
                self.matchRange(88, 90)






        finally:

            pass

    # $ANTLR end "HGLETTERALL"



    # $ANTLR start "LETTER"
    def mLETTER(self, ):

        try:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:140:13: ( LWLETTER | HGLETTER )
            alt2 = 2
            LA2_0 = self.input.LA(1)

            if ((97 <= LA2_0 <= 122)) :
                alt2 = 1
            elif ((65 <= LA2_0 <= 90)) :
                alt2 = 2
            else:
                nvae = NoViableAltException("", 2, 0, self.input)

                raise nvae

            if alt2 == 1:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:140:15: LWLETTER
                pass 
                self.mLWLETTER()


            elif alt2 == 2:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:140:26: HGLETTER
                pass 
                self.mHGLETTER()



        finally:

            pass

    # $ANTLR end "LETTER"



    # $ANTLR start "INT"
    def mINT(self, ):

        try:
            _type = INT
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:5: ( ( '0' | ( ( '1' .. '9' ) ( DIGIT )* ) ) )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:7: ( '0' | ( ( '1' .. '9' ) ( DIGIT )* ) )
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:7: ( '0' | ( ( '1' .. '9' ) ( DIGIT )* ) )
            alt4 = 2
            LA4_0 = self.input.LA(1)

            if (LA4_0 == 48) :
                alt4 = 1
            elif ((49 <= LA4_0 <= 57)) :
                alt4 = 2
            else:
                nvae = NoViableAltException("", 4, 0, self.input)

                raise nvae

            if alt4 == 1:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:8: '0'
                pass 
                self.match(48)


            elif alt4 == 2:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:14: ( ( '1' .. '9' ) ( DIGIT )* )
                pass 
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:14: ( ( '1' .. '9' ) ( DIGIT )* )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:15: ( '1' .. '9' ) ( DIGIT )*
                pass 
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:15: ( '1' .. '9' )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:16: '1' .. '9'
                pass 
                self.matchRange(49, 57)



                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:25: ( DIGIT )*
                while True: #loop3
                    alt3 = 2
                    LA3_0 = self.input.LA(1)

                    if ((48 <= LA3_0 <= 57)) :
                        alt3 = 1


                    if alt3 == 1:
                        # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:144:25: DIGIT
                        pass 
                        self.mDIGIT()


                    else:
                        break #loop3









            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "INT"



    # $ANTLR start "TRUE"
    def mTRUE(self, ):

        try:
            _type = TRUE
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:148:6: ( 'True' | 'true' )
            alt5 = 2
            LA5_0 = self.input.LA(1)

            if (LA5_0 == 84) :
                alt5 = 1
            elif (LA5_0 == 116) :
                alt5 = 2
            else:
                nvae = NoViableAltException("", 5, 0, self.input)

                raise nvae

            if alt5 == 1:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:148:8: 'True'
                pass 
                self.match("True")


            elif alt5 == 2:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:148:17: 'true'
                pass 
                self.match("true")


            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "TRUE"



    # $ANTLR start "FALSE"
    def mFALSE(self, ):

        try:
            _type = FALSE
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:149:7: ( 'False' | 'false' )
            alt6 = 2
            LA6_0 = self.input.LA(1)

            if (LA6_0 == 70) :
                alt6 = 1
            elif (LA6_0 == 102) :
                alt6 = 2
            else:
                nvae = NoViableAltException("", 6, 0, self.input)

                raise nvae

            if alt6 == 1:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:149:9: 'False'
                pass 
                self.match("False")


            elif alt6 == 2:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:149:19: 'false'
                pass 
                self.match("false")


            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "FALSE"



    # $ANTLR start "PROP"
    def mPROP(self, ):

        try:
            _type = PROP
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:7: ( ( ( LWLETTER | HGLETTERALL ) ( '_' | LETTER | DIGIT )* ) )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:9: ( ( LWLETTER | HGLETTERALL ) ( '_' | LETTER | DIGIT )* )
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:9: ( ( LWLETTER | HGLETTERALL ) ( '_' | LETTER | DIGIT )* )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:10: ( LWLETTER | HGLETTERALL ) ( '_' | LETTER | DIGIT )*
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:10: ( LWLETTER | HGLETTERALL )
            alt7 = 2
            LA7_0 = self.input.LA(1)

            if ((97 <= LA7_0 <= 122)) :
                alt7 = 1
            elif ((65 <= LA7_0 <= 71) or (73 <= LA7_0 <= 86) or (88 <= LA7_0 <= 90)) :
                alt7 = 2
            else:
                nvae = NoViableAltException("", 7, 0, self.input)

                raise nvae

            if alt7 == 1:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:11: LWLETTER
                pass 
                self.mLWLETTER()


            elif alt7 == 2:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:22: HGLETTERALL
                pass 
                self.mHGLETTERALL()



            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:34: ( '_' | LETTER | DIGIT )*
            while True: #loop8
                alt8 = 4
                LA8 = self.input.LA(1)
                if LA8 == 95:
                    alt8 = 1
                elif LA8 == 65 or LA8 == 66 or LA8 == 67 or LA8 == 68 or LA8 == 69 or LA8 == 70 or LA8 == 71 or LA8 == 72 or LA8 == 73 or LA8 == 74 or LA8 == 75 or LA8 == 76 or LA8 == 77 or LA8 == 78 or LA8 == 79 or LA8 == 80 or LA8 == 81 or LA8 == 82 or LA8 == 83 or LA8 == 84 or LA8 == 85 or LA8 == 86 or LA8 == 87 or LA8 == 88 or LA8 == 89 or LA8 == 90 or LA8 == 97 or LA8 == 98 or LA8 == 99 or LA8 == 100 or LA8 == 101 or LA8 == 102 or LA8 == 103 or LA8 == 104 or LA8 == 105 or LA8 == 106 or LA8 == 107 or LA8 == 108 or LA8 == 109 or LA8 == 110 or LA8 == 111 or LA8 == 112 or LA8 == 113 or LA8 == 114 or LA8 == 115 or LA8 == 116 or LA8 == 117 or LA8 == 118 or LA8 == 119 or LA8 == 120 or LA8 == 121 or LA8 == 122:
                    alt8 = 2
                elif LA8 == 48 or LA8 == 49 or LA8 == 50 or LA8 == 51 or LA8 == 52 or LA8 == 53 or LA8 == 54 or LA8 == 55 or LA8 == 56 or LA8 == 57:
                    alt8 = 3

                if alt8 == 1:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:35: '_'
                    pass 
                    self.match(95)


                elif alt8 == 2:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:41: LETTER
                    pass 
                    self.mLETTER()


                elif alt8 == 3:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:152:50: DIGIT
                    pass 
                    self.mDIGIT()


                else:
                    break #loop8



            #action start
                 
            if str(self.text).lower() not in ('true', 'false'):
                self.alphabet.add(str(self.text));
                
            #action end



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "PROP"



    # $ANTLR start "LINECMT"
    def mLINECMT(self, ):

        try:
            _type = LINECMT
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:160:9: ( ( '//' ) (~ ( '\\n' | '\\r' ) )* )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:160:11: ( '//' ) (~ ( '\\n' | '\\r' ) )*
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:160:11: ( '//' )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:160:12: '//'
            pass 
            self.match("//")



            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:160:17: (~ ( '\\n' | '\\r' ) )*
            while True: #loop9
                alt9 = 2
                LA9_0 = self.input.LA(1)

                if ((0 <= LA9_0 <= 9) or (11 <= LA9_0 <= 12) or (14 <= LA9_0 <= 65535)) :
                    alt9 = 1


                if alt9 == 1:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:160:18: ~ ( '\\n' | '\\r' )
                    pass 
                    if (0 <= self.input.LA(1) <= 9) or (11 <= self.input.LA(1) <= 12) or (14 <= self.input.LA(1) <= 65535):
                        self.input.consume()
                    else:
                        mse = MismatchedSetException(None, self.input)
                        self.recover(mse)
                        raise mse



                else:
                    break #loop9
            #action start
            self.skip()
            #action end



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "LINECMT"



    # $ANTLR start "WS"
    def mWS(self, ):

        try:
            _type = WS
            _channel = DEFAULT_CHANNEL

            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:165:5: ( ( ( '\\n' | '\\r' | '\\f' | '\\t' | ' ' )+ ) )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:165:7: ( ( '\\n' | '\\r' | '\\f' | '\\t' | ' ' )+ )
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:165:7: ( ( '\\n' | '\\r' | '\\f' | '\\t' | ' ' )+ )
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:165:8: ( '\\n' | '\\r' | '\\f' | '\\t' | ' ' )+
            pass 
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:165:8: ( '\\n' | '\\r' | '\\f' | '\\t' | ' ' )+
            cnt10 = 0
            while True: #loop10
                alt10 = 2
                LA10_0 = self.input.LA(1)

                if ((9 <= LA10_0 <= 10) or (12 <= LA10_0 <= 13) or LA10_0 == 32) :
                    alt10 = 1


                if alt10 == 1:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:
                    pass 
                    if (9 <= self.input.LA(1) <= 10) or (12 <= self.input.LA(1) <= 13) or self.input.LA(1) == 32:
                        self.input.consume()
                    else:
                        mse = MismatchedSetException(None, self.input)
                        self.recover(mse)
                        raise mse



                else:
                    if cnt10 >= 1:
                        break #loop10

                    eee = EarlyExitException(10, self.input)
                    raise eee

                cnt10 += 1



            #action start
            self.skip()
            #action end



            self._state.type = _type
            self._state.channel = _channel

        finally:

            pass

    # $ANTLR end "WS"



    def mTokens(self):
        # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:8: ( AND | OR | NOT | HOLD | WITHIN | CONCAT | T__21 | T__22 | T__23 | T__24 | T__25 | T__26 | INT | TRUE | FALSE | PROP | LINECMT | WS )
        alt11 = 18
        alt11 = self.dfa11.predict(self.input)
        if alt11 == 1:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:10: AND
            pass 
            self.mAND()


        elif alt11 == 2:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:14: OR
            pass 
            self.mOR()


        elif alt11 == 3:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:17: NOT
            pass 
            self.mNOT()


        elif alt11 == 4:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:21: HOLD
            pass 
            self.mHOLD()


        elif alt11 == 5:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:26: WITHIN
            pass 
            self.mWITHIN()


        elif alt11 == 6:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:33: CONCAT
            pass 
            self.mCONCAT()


        elif alt11 == 7:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:40: T__21
            pass 
            self.mT__21()


        elif alt11 == 8:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:46: T__22
            pass 
            self.mT__22()


        elif alt11 == 9:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:52: T__23
            pass 
            self.mT__23()


        elif alt11 == 10:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:58: T__24
            pass 
            self.mT__24()


        elif alt11 == 11:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:64: T__25
            pass 
            self.mT__25()


        elif alt11 == 12:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:70: T__26
            pass 
            self.mT__26()


        elif alt11 == 13:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:76: INT
            pass 
            self.mINT()


        elif alt11 == 14:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:80: TRUE
            pass 
            self.mTRUE()


        elif alt11 == 15:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:85: FALSE
            pass 
            self.mFALSE()


        elif alt11 == 16:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:91: PROP
            pass 
            self.mPROP()


        elif alt11 == 17:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:96: LINECMT
            pass 
            self.mLINECMT()


        elif alt11 == 18:
            # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:1:104: WS
            pass 
            self.mWS()







    # lookup tables for DFA #11

    DFA11_eot = DFA.unpack(
        u"\16\uffff\4\22\3\uffff\10\22\2\41\2\22\1\uffff\2\44\1\uffff"
        )

    DFA11_eof = DFA.unpack(
        u"\45\uffff"
        )

    DFA11_min = DFA.unpack(
        u"\1\11\15\uffff\2\162\2\141\3\uffff\2\165\2\154\2\145\2\163\2\60"
        u"\2\145\1\uffff\2\60\1\uffff"
        )

    DFA11_max = DFA.unpack(
        u"\1\174\15\uffff\2\162\2\141\3\uffff\2\165\2\154\2\145\2\163\2"
        u"\172\2\145\1\uffff\2\172\1\uffff"
        )

    DFA11_accept = DFA.unpack(
        u"\1\uffff\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\12\1\13\1\14"
        u"\1\15\4\uffff\1\20\1\21\1\22\14\uffff\1\16\2\uffff\1\17"
        )

    DFA11_special = DFA.unpack(
        u"\45\uffff"
        )

            
    DFA11_transition = [
        DFA.unpack(u"\2\24\1\uffff\2\24\22\uffff\1\24\1\3\4\uffff\1\1\1"
        u"\uffff\1\13\1\14\1\6\1\uffff\1\12\2\uffff\1\23\12\15\7\uffff\5"
        u"\22\1\20\1\22\1\4\13\22\1\16\2\22\1\5\3\22\1\10\1\uffff\1\11\1"
        u"\7\2\uffff\5\22\1\21\15\22\1\17\6\22\1\uffff\1\2"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\1\25"),
        DFA.unpack(u"\1\26"),
        DFA.unpack(u"\1\27"),
        DFA.unpack(u"\1\30"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\1\31"),
        DFA.unpack(u"\1\32"),
        DFA.unpack(u"\1\33"),
        DFA.unpack(u"\1\34"),
        DFA.unpack(u"\1\35"),
        DFA.unpack(u"\1\36"),
        DFA.unpack(u"\1\37"),
        DFA.unpack(u"\1\40"),
        DFA.unpack(u"\12\22\7\uffff\32\22\4\uffff\1\22\1\uffff\32\22"),
        DFA.unpack(u"\12\22\7\uffff\32\22\4\uffff\1\22\1\uffff\32\22"),
        DFA.unpack(u"\1\42"),
        DFA.unpack(u"\1\43"),
        DFA.unpack(u""),
        DFA.unpack(u"\12\22\7\uffff\32\22\4\uffff\1\22\1\uffff\32\22"),
        DFA.unpack(u"\12\22\7\uffff\32\22\4\uffff\1\22\1\uffff\32\22"),
        DFA.unpack(u"")
    ]

    # class definition for DFA #11

    class DFA11(DFA):
        pass


 



def main(argv, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr):
    from antlr3.main import LexerMain
    main = LexerMain(twtlLexer)
    main.stdin = stdin
    main.stdout = stdout
    main.stderr = stderr
    main.execute(argv)


if __name__ == '__main__':
    main(sys.argv)
