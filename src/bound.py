# $ANTLR 3.1.3 Mar 18, 2009 10:09:25 C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g 2016-02-22 17:58:48

import sys
from antlr3 import *
from antlr3.tree import *
from antlr3.compat import set, frozenset
         
license_text='''
    Module computes the bound of a TWTL formula. 
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

# token names
tokenNames = [
    "<invalid>", "<EOR>", "<DOWN>", "<UP>", 
    "AND", "OR", "NOT", "HOLD", "WITHIN", "CONCAT", "INT", "PROP", "TRUE", 
    "FALSE", "DIGIT", "LWLETTER", "HGLETTER", "HGLETTERALL", "LETTER", "LINECMT", 
    "WS", "'^'", "'['", "']'", "','", "'('", "')'"
]




class bound(TreeParser):
    grammarFileName = "C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g"
    antlr_version = version_str_to_tuple("3.1.3 Mar 18, 2009 10:09:25")
    antlr_version_str = "3.1.3 Mar 18, 2009 10:09:25"
    tokenNames = tokenNames

    def __init__(self, input, state=None, *args, **kwargs):
        if state is None:
            state = RecognizerSharedState()

        super(bound, self).__init__(input, state, *args, **kwargs)

        self.dfa1 = self.DFA1(
            self, 1,
            eot = self.DFA1_eot,
            eof = self.DFA1_eof,
            min = self.DFA1_min,
            max = self.DFA1_max,
            accept = self.DFA1_accept,
            special = self.DFA1_special,
            transition = self.DFA1_transition
            )






                


        

              
    bound = 0

    def getBound(self):
        return self.bound



    # $ANTLR start "eval"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:58:1: eval : formula ;
    def eval(self, ):

        formula1 = None


        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:58:5: ( formula )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:58:9: formula
                pass 
                self._state.following.append(self.FOLLOW_formula_in_eval62)
                formula1 = self.formula()

                self._state.following.pop()
                #action start
                self.bound = formula1;
                #action end




            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
        finally:

            pass
        return 

    # $ANTLR end "eval"


    # $ANTLR start "formula"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:61:1: formula returns [value] : ( ^( OR a= formula b= formula ) | ^( AND a= formula b= formula ) | ^( NOT a= formula ) | ^( CONCAT a= formula b= formula ) | ^( HOLD INT PROP ) | ^( HOLD INT ^( NOT PROP ) ) | ^( WITHIN phi= formula low= INT high= INT ) | PROP | TRUE | FALSE );
    def formula(self, ):

        value = None

        low = None
        high = None
        INT2 = None
        INT3 = None
        a = None

        b = None

        phi = None


        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:62:5: ( ^( OR a= formula b= formula ) | ^( AND a= formula b= formula ) | ^( NOT a= formula ) | ^( CONCAT a= formula b= formula ) | ^( HOLD INT PROP ) | ^( HOLD INT ^( NOT PROP ) ) | ^( WITHIN phi= formula low= INT high= INT ) | PROP | TRUE | FALSE )
                alt1 = 10
                alt1 = self.dfa1.predict(self.input)
                if alt1 == 1:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:62:9: ^( OR a= formula b= formula )
                    pass 
                    self.match(self.input, OR, self.FOLLOW_OR_in_formula88)

                    self.match(self.input, DOWN, None)
                    self._state.following.append(self.FOLLOW_formula_in_formula93)
                    a = self.formula()

                    self._state.following.pop()
                    self._state.following.append(self.FOLLOW_formula_in_formula97)
                    b = self.formula()

                    self._state.following.pop()

                    self.match(self.input, UP, None)
                    #action start
                    value = (min(a[0], b[0]), max(a[1], b[1]))
                    #action end


                elif alt1 == 2:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:63:9: ^( AND a= formula b= formula )
                    pass 
                    self.match(self.input, AND, self.FOLLOW_AND_in_formula111)

                    self.match(self.input, DOWN, None)
                    self._state.following.append(self.FOLLOW_formula_in_formula115)
                    a = self.formula()

                    self._state.following.pop()
                    self._state.following.append(self.FOLLOW_formula_in_formula119)
                    b = self.formula()

                    self._state.following.pop()

                    self.match(self.input, UP, None)
                    #action start
                    value = (max(a[0], b[0]), max(a[1], b[1]))
                    #action end


                elif alt1 == 3:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:64:9: ^( NOT a= formula )
                    pass 
                    self.match(self.input, NOT, self.FOLLOW_NOT_in_formula133)

                    self.match(self.input, DOWN, None)
                    self._state.following.append(self.FOLLOW_formula_in_formula137)
                    a = self.formula()

                    self._state.following.pop()

                    self.match(self.input, UP, None)
                    #action start
                    value = a
                    #action end


                elif alt1 == 4:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:65:9: ^( CONCAT a= formula b= formula )
                    pass 
                    self.match(self.input, CONCAT, self.FOLLOW_CONCAT_in_formula161)

                    self.match(self.input, DOWN, None)
                    self._state.following.append(self.FOLLOW_formula_in_formula165)
                    a = self.formula()

                    self._state.following.pop()
                    self._state.following.append(self.FOLLOW_formula_in_formula169)
                    b = self.formula()

                    self._state.following.pop()

                    self.match(self.input, UP, None)
                    #action start
                    value = [x+y+1 for x, y in zip(a, b)]
                    #action end


                elif alt1 == 5:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:66:9: ^( HOLD INT PROP )
                    pass 
                    self.match(self.input, HOLD, self.FOLLOW_HOLD_in_formula183)

                    self.match(self.input, DOWN, None)
                    INT2=self.match(self.input, INT, self.FOLLOW_INT_in_formula185)
                    self.match(self.input, PROP, self.FOLLOW_PROP_in_formula187)

                    self.match(self.input, UP, None)
                    #action start
                    value = (int(INT2.text), int(INT2.text))
                    #action end


                elif alt1 == 6:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:67:9: ^( HOLD INT ^( NOT PROP ) )
                    pass 
                    self.match(self.input, HOLD, self.FOLLOW_HOLD_in_formula208)

                    self.match(self.input, DOWN, None)
                    INT3=self.match(self.input, INT, self.FOLLOW_INT_in_formula210)
                    self.match(self.input, NOT, self.FOLLOW_NOT_in_formula213)

                    self.match(self.input, DOWN, None)
                    self.match(self.input, PROP, self.FOLLOW_PROP_in_formula215)

                    self.match(self.input, UP, None)

                    self.match(self.input, UP, None)
                    #action start
                    value = (int(INT3.text), int(INT3.text))
                    #action end


                elif alt1 == 7:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:68:9: ^( WITHIN phi= formula low= INT high= INT )
                    pass 
                    self.match(self.input, WITHIN, self.FOLLOW_WITHIN_in_formula230)

                    self.match(self.input, DOWN, None)
                    self._state.following.append(self.FOLLOW_formula_in_formula234)
                    phi = self.formula()

                    self._state.following.pop()
                    low=self.match(self.input, INT, self.FOLLOW_INT_in_formula238)
                    high=self.match(self.input, INT, self.FOLLOW_INT_in_formula242)

                    self.match(self.input, UP, None)
                    #action start
                                                                    
                    value = (int(low.text)+phi[0], int(high.text)) # - int(low.text)
                    if phi[1] > int(high.text) - int(low.text):
                        raise ValueError("Within operator deadline is invalid!")
                            
                    #action end


                elif alt1 == 8:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:73:9: PROP
                    pass 
                    self.match(self.input, PROP, self.FOLLOW_PROP_in_formula255)
                    #action start
                    value =  (0, 0)
                    #action end


                elif alt1 == 9:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:74:9: TRUE
                    pass 
                    self.match(self.input, TRUE, self.FOLLOW_TRUE_in_formula290)
                    #action start
                    value =  (0, 0)
                    #action end


                elif alt1 == 10:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\bound.g:75:9: FALSE
                    pass 
                    self.match(self.input, FALSE, self.FOLLOW_FALSE_in_formula325)
                    #action start
                    value =  (0, 0)
                    #action end



            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
        finally:

            pass
        return value

    # $ANTLR end "formula"


    # Delegated rules


    # lookup tables for DFA #1

    DFA1_eot = DFA.unpack(
        u"\16\uffff"
        )

    DFA1_eof = DFA.unpack(
        u"\16\uffff"
        )

    DFA1_min = DFA.unpack(
        u"\1\4\4\uffff\1\2\4\uffff\1\12\1\6\2\uffff"
        )

    DFA1_max = DFA.unpack(
        u"\1\15\4\uffff\1\2\4\uffff\1\12\1\13\2\uffff"
        )

    DFA1_accept = DFA.unpack(
        u"\1\uffff\1\1\1\2\1\3\1\4\1\uffff\1\7\1\10\1\11\1\12\2\uffff\1"
        u"\5\1\6"
        )

    DFA1_special = DFA.unpack(
        u"\16\uffff"
        )

            
    DFA1_transition = [
        DFA.unpack(u"\1\2\1\1\1\3\1\5\1\6\1\4\1\uffff\1\7\1\10\1\11"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\1\12"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\1\13"),
        DFA.unpack(u"\1\15\4\uffff\1\14"),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #1

    class DFA1(DFA):
        pass


 

    FOLLOW_formula_in_eval62 = frozenset([1])
    FOLLOW_OR_in_formula88 = frozenset([2])
    FOLLOW_formula_in_formula93 = frozenset([4, 5, 6, 7, 8, 9, 11, 12, 13])
    FOLLOW_formula_in_formula97 = frozenset([3])
    FOLLOW_AND_in_formula111 = frozenset([2])
    FOLLOW_formula_in_formula115 = frozenset([4, 5, 6, 7, 8, 9, 11, 12, 13])
    FOLLOW_formula_in_formula119 = frozenset([3])
    FOLLOW_NOT_in_formula133 = frozenset([2])
    FOLLOW_formula_in_formula137 = frozenset([3])
    FOLLOW_CONCAT_in_formula161 = frozenset([2])
    FOLLOW_formula_in_formula165 = frozenset([4, 5, 6, 7, 8, 9, 11, 12, 13])
    FOLLOW_formula_in_formula169 = frozenset([3])
    FOLLOW_HOLD_in_formula183 = frozenset([2])
    FOLLOW_INT_in_formula185 = frozenset([11])
    FOLLOW_PROP_in_formula187 = frozenset([3])
    FOLLOW_HOLD_in_formula208 = frozenset([2])
    FOLLOW_INT_in_formula210 = frozenset([6])
    FOLLOW_NOT_in_formula213 = frozenset([2])
    FOLLOW_PROP_in_formula215 = frozenset([3])
    FOLLOW_WITHIN_in_formula230 = frozenset([2])
    FOLLOW_formula_in_formula234 = frozenset([10])
    FOLLOW_INT_in_formula238 = frozenset([10])
    FOLLOW_INT_in_formula242 = frozenset([3])
    FOLLOW_PROP_in_formula255 = frozenset([1])
    FOLLOW_TRUE_in_formula290 = frozenset([1])
    FOLLOW_FALSE_in_formula325 = frozenset([1])



def main(argv, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr):
    from antlr3.main import WalkerMain
    main = WalkerMain(bound)
    main.stdin = stdin
    main.stdout = stdout
    main.stderr = stderr
    main.execute(argv)


if __name__ == '__main__':
    main(sys.argv)
