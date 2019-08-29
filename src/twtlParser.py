# $ANTLR 3.1.3 Mar 18, 2009 10:09:25 C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g 2016-02-22 17:58:47

import sys
from antlr3 import *
from antlr3.compat import set, frozenset

from antlr3.tree import *

         
license_text='''
    Parser for TWTL formulae. 
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




class twtlParser(Parser):
    grammarFileName = "C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g"
    antlr_version = version_str_to_tuple("3.1.3 Mar 18, 2009 10:09:25")
    antlr_version_str = "3.1.3 Mar 18, 2009 10:09:25"
    tokenNames = tokenNames

    def __init__(self, input, state=None, *args, **kwargs):
        if state is None:
            state = RecognizerSharedState()

        super(twtlParser, self).__init__(input, state, *args, **kwargs)






        self._adaptor = None
        self.adaptor = CommonTreeAdaptor()
                


        
    def getTreeAdaptor(self):
        return self._adaptor

    def setTreeAdaptor(self, adaptor):
        self._adaptor = adaptor

    adaptor = property(getTreeAdaptor, setTreeAdaptor)

              


    class prog_return(ParserRuleReturnScope):
        def __init__(self):
            super(twtlParser.prog_return, self).__init__()

            self.tree = None




    # $ANTLR start "prog"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:96:1: prog : formula ;
    def prog(self, ):

        retval = self.prog_return()
        retval.start = self.input.LT(1)

        root_0 = None

        formula1 = None



        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:96:5: ( formula )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:96:9: formula
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_formula_in_prog171)
                formula1 = self.formula()

                self._state.following.pop()
                self._adaptor.addChild(root_0, formula1.tree)



                retval.stop = self.input.LT(-1)


                retval.tree = self._adaptor.rulePostProcessing(root_0)
                self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:

            pass
        return retval

    # $ANTLR end "prog"

    class formula_return(ParserRuleReturnScope):
        def __init__(self):
            super(twtlParser.formula_return, self).__init__()

            self.tree = None




    # $ANTLR start "formula"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:98:1: formula : disjunction ;
    def formula(self, ):

        retval = self.formula_return()
        retval.start = self.input.LT(1)

        root_0 = None

        disjunction2 = None



        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:98:9: ( disjunction )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:98:13: disjunction
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_disjunction_in_formula182)
                disjunction2 = self.disjunction()

                self._state.following.pop()
                self._adaptor.addChild(root_0, disjunction2.tree)



                retval.stop = self.input.LT(-1)


                retval.tree = self._adaptor.rulePostProcessing(root_0)
                self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:

            pass
        return retval

    # $ANTLR end "formula"

    class disjunction_return(ParserRuleReturnScope):
        def __init__(self):
            super(twtlParser.disjunction_return, self).__init__()

            self.tree = None




    # $ANTLR start "disjunction"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:100:1: disjunction : conjunction ( OR conjunction )* ;
    def disjunction(self, ):

        retval = self.disjunction_return()
        retval.start = self.input.LT(1)

        root_0 = None

        OR4 = None
        conjunction3 = None

        conjunction5 = None


        OR4_tree = None

        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:101:5: ( conjunction ( OR conjunction )* )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:101:7: conjunction ( OR conjunction )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_conjunction_in_disjunction195)
                conjunction3 = self.conjunction()

                self._state.following.pop()
                self._adaptor.addChild(root_0, conjunction3.tree)
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:101:19: ( OR conjunction )*
                while True: #loop1
                    alt1 = 2
                    LA1_0 = self.input.LA(1)

                    if (LA1_0 == OR) :
                        alt1 = 1


                    if alt1 == 1:
                        # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:101:20: OR conjunction
                        pass 
                        OR4=self.match(self.input, OR, self.FOLLOW_OR_in_disjunction198)

                        OR4_tree = self._adaptor.createWithPayload(OR4)
                        root_0 = self._adaptor.becomeRoot(OR4_tree, root_0)

                        self._state.following.append(self.FOLLOW_conjunction_in_disjunction201)
                        conjunction5 = self.conjunction()

                        self._state.following.pop()
                        self._adaptor.addChild(root_0, conjunction5.tree)


                    else:
                        break #loop1



                retval.stop = self.input.LT(-1)


                retval.tree = self._adaptor.rulePostProcessing(root_0)
                self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:

            pass
        return retval

    # $ANTLR end "disjunction"

    class conjunction_return(ParserRuleReturnScope):
        def __init__(self):
            super(twtlParser.conjunction_return, self).__init__()

            self.tree = None




    # $ANTLR start "conjunction"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:104:1: conjunction : concatenation ( AND concatenation )* ;
    def conjunction(self, ):

        retval = self.conjunction_return()
        retval.start = self.input.LT(1)

        root_0 = None

        AND7 = None
        concatenation6 = None

        concatenation8 = None


        AND7_tree = None

        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:105:5: ( concatenation ( AND concatenation )* )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:105:9: concatenation ( AND concatenation )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_concatenation_in_conjunction224)
                concatenation6 = self.concatenation()

                self._state.following.pop()
                self._adaptor.addChild(root_0, concatenation6.tree)
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:105:23: ( AND concatenation )*
                while True: #loop2
                    alt2 = 2
                    LA2_0 = self.input.LA(1)

                    if (LA2_0 == AND) :
                        alt2 = 1


                    if alt2 == 1:
                        # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:105:24: AND concatenation
                        pass 
                        AND7=self.match(self.input, AND, self.FOLLOW_AND_in_conjunction227)

                        AND7_tree = self._adaptor.createWithPayload(AND7)
                        root_0 = self._adaptor.becomeRoot(AND7_tree, root_0)

                        self._state.following.append(self.FOLLOW_concatenation_in_conjunction230)
                        concatenation8 = self.concatenation()

                        self._state.following.pop()
                        self._adaptor.addChild(root_0, concatenation8.tree)


                    else:
                        break #loop2



                retval.stop = self.input.LT(-1)


                retval.tree = self._adaptor.rulePostProcessing(root_0)
                self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:

            pass
        return retval

    # $ANTLR end "conjunction"

    class concatenation_return(ParserRuleReturnScope):
        def __init__(self):
            super(twtlParser.concatenation_return, self).__init__()

            self.tree = None




    # $ANTLR start "concatenation"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:108:1: concatenation : temporal ( CONCAT temporal )* ;
    def concatenation(self, ):

        retval = self.concatenation_return()
        retval.start = self.input.LT(1)

        root_0 = None

        CONCAT10 = None
        temporal9 = None

        temporal11 = None


        CONCAT10_tree = None

        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:109:5: ( temporal ( CONCAT temporal )* )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:109:9: temporal ( CONCAT temporal )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_temporal_in_concatenation252)
                temporal9 = self.temporal()

                self._state.following.pop()
                self._adaptor.addChild(root_0, temporal9.tree)
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:109:18: ( CONCAT temporal )*
                while True: #loop3
                    alt3 = 2
                    LA3_0 = self.input.LA(1)

                    if (LA3_0 == CONCAT) :
                        alt3 = 1


                    if alt3 == 1:
                        # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:109:19: CONCAT temporal
                        pass 
                        CONCAT10=self.match(self.input, CONCAT, self.FOLLOW_CONCAT_in_concatenation255)

                        CONCAT10_tree = self._adaptor.createWithPayload(CONCAT10)
                        root_0 = self._adaptor.becomeRoot(CONCAT10_tree, root_0)

                        self._state.following.append(self.FOLLOW_temporal_in_concatenation258)
                        temporal11 = self.temporal()

                        self._state.following.pop()
                        self._adaptor.addChild(root_0, temporal11.tree)


                    else:
                        break #loop3



                retval.stop = self.input.LT(-1)


                retval.tree = self._adaptor.rulePostProcessing(root_0)
                self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:

            pass
        return retval

    # $ANTLR end "concatenation"

    class temporal_return(ParserRuleReturnScope):
        def __init__(self):
            super(twtlParser.temporal_return, self).__init__()

            self.tree = None




    # $ANTLR start "temporal"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:112:1: temporal : ( HOLD '^' duration= INT PROP | HOLD '^' duration= INT NOT p= PROP -> ^( HOLD $duration ^( NOT $p) ) | ( '[' phi= formula ']' '^' '[' low= INT ',' high= INT ']' ) -> ^( WITHIN['W'] $phi $low $high) | negation );
    def temporal(self, ):

        retval = self.temporal_return()
        retval.start = self.input.LT(1)

        root_0 = None

        duration = None
        p = None
        low = None
        high = None
        HOLD12 = None
        char_literal13 = None
        PROP14 = None
        HOLD15 = None
        char_literal16 = None
        NOT17 = None
        char_literal18 = None
        char_literal19 = None
        char_literal20 = None
        char_literal21 = None
        char_literal22 = None
        char_literal23 = None
        phi = None

        negation24 = None


        duration_tree = None
        p_tree = None
        low_tree = None
        high_tree = None
        HOLD12_tree = None
        char_literal13_tree = None
        PROP14_tree = None
        HOLD15_tree = None
        char_literal16_tree = None
        NOT17_tree = None
        char_literal18_tree = None
        char_literal19_tree = None
        char_literal20_tree = None
        char_literal21_tree = None
        char_literal22_tree = None
        char_literal23_tree = None
        stream_22 = RewriteRuleTokenStream(self._adaptor, "token 22")
        stream_23 = RewriteRuleTokenStream(self._adaptor, "token 23")
        stream_NOT = RewriteRuleTokenStream(self._adaptor, "token NOT")
        stream_24 = RewriteRuleTokenStream(self._adaptor, "token 24")
        stream_PROP = RewriteRuleTokenStream(self._adaptor, "token PROP")
        stream_HOLD = RewriteRuleTokenStream(self._adaptor, "token HOLD")
        stream_INT = RewriteRuleTokenStream(self._adaptor, "token INT")
        stream_21 = RewriteRuleTokenStream(self._adaptor, "token 21")
        stream_formula = RewriteRuleSubtreeStream(self._adaptor, "rule formula")
        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:113:5: ( HOLD '^' duration= INT PROP | HOLD '^' duration= INT NOT p= PROP -> ^( HOLD $duration ^( NOT $p) ) | ( '[' phi= formula ']' '^' '[' low= INT ',' high= INT ']' ) -> ^( WITHIN['W'] $phi $low $high) | negation )
                alt4 = 4
                LA4 = self.input.LA(1)
                if LA4 == HOLD:
                    LA4_1 = self.input.LA(2)

                    if (LA4_1 == 21) :
                        LA4_4 = self.input.LA(3)

                        if (LA4_4 == INT) :
                            LA4_5 = self.input.LA(4)

                            if (LA4_5 == PROP) :
                                alt4 = 1
                            elif (LA4_5 == NOT) :
                                alt4 = 2
                            else:
                                nvae = NoViableAltException("", 4, 5, self.input)

                                raise nvae

                        else:
                            nvae = NoViableAltException("", 4, 4, self.input)

                            raise nvae

                    else:
                        nvae = NoViableAltException("", 4, 1, self.input)

                        raise nvae

                elif LA4 == 22:
                    alt4 = 3
                elif LA4 == NOT or LA4 == PROP or LA4 == TRUE or LA4 == FALSE or LA4 == 25:
                    alt4 = 4
                else:
                    nvae = NoViableAltException("", 4, 0, self.input)

                    raise nvae

                if alt4 == 1:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:113:7: HOLD '^' duration= INT PROP
                    pass 
                    root_0 = self._adaptor.nil()

                    HOLD12=self.match(self.input, HOLD, self.FOLLOW_HOLD_in_temporal278)

                    HOLD12_tree = self._adaptor.createWithPayload(HOLD12)
                    root_0 = self._adaptor.becomeRoot(HOLD12_tree, root_0)

                    char_literal13=self.match(self.input, 21, self.FOLLOW_21_in_temporal281)
                    duration=self.match(self.input, INT, self.FOLLOW_INT_in_temporal286)

                    duration_tree = self._adaptor.createWithPayload(duration)
                    self._adaptor.addChild(root_0, duration_tree)

                    PROP14=self.match(self.input, PROP, self.FOLLOW_PROP_in_temporal288)

                    PROP14_tree = self._adaptor.createWithPayload(PROP14)
                    self._adaptor.addChild(root_0, PROP14_tree)



                elif alt4 == 2:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:114:7: HOLD '^' duration= INT NOT p= PROP
                    pass 
                    HOLD15=self.match(self.input, HOLD, self.FOLLOW_HOLD_in_temporal296) 
                    stream_HOLD.add(HOLD15)
                    char_literal16=self.match(self.input, 21, self.FOLLOW_21_in_temporal298) 
                    stream_21.add(char_literal16)
                    duration=self.match(self.input, INT, self.FOLLOW_INT_in_temporal302) 
                    stream_INT.add(duration)
                    NOT17=self.match(self.input, NOT, self.FOLLOW_NOT_in_temporal304) 
                    stream_NOT.add(NOT17)
                    p=self.match(self.input, PROP, self.FOLLOW_PROP_in_temporal308) 
                    stream_PROP.add(p)

                    # AST Rewrite
                    # elements: HOLD, p, duration, NOT
                    # token labels: p, duration
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 

                    retval.tree = root_0
                    stream_p = RewriteRuleTokenStream(self._adaptor, "token p", p)
                    stream_duration = RewriteRuleTokenStream(self._adaptor, "token duration", duration)

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 114:40: -> ^( HOLD $duration ^( NOT $p) )
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:114:43: ^( HOLD $duration ^( NOT $p) )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(stream_HOLD.nextNode(), root_1)

                    self._adaptor.addChild(root_1, stream_duration.nextNode())
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:114:60: ^( NOT $p)
                    root_2 = self._adaptor.nil()
                    root_2 = self._adaptor.becomeRoot(stream_NOT.nextNode(), root_2)

                    self._adaptor.addChild(root_2, stream_p.nextNode())

                    self._adaptor.addChild(root_1, root_2)

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0


                elif alt4 == 3:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:115:7: ( '[' phi= formula ']' '^' '[' low= INT ',' high= INT ']' )
                    pass 
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:115:7: ( '[' phi= formula ']' '^' '[' low= INT ',' high= INT ']' )
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:115:8: '[' phi= formula ']' '^' '[' low= INT ',' high= INT ']'
                    pass 
                    char_literal18=self.match(self.input, 22, self.FOLLOW_22_in_temporal333) 
                    stream_22.add(char_literal18)
                    self._state.following.append(self.FOLLOW_formula_in_temporal337)
                    phi = self.formula()

                    self._state.following.pop()
                    stream_formula.add(phi.tree)
                    char_literal19=self.match(self.input, 23, self.FOLLOW_23_in_temporal339) 
                    stream_23.add(char_literal19)
                    char_literal20=self.match(self.input, 21, self.FOLLOW_21_in_temporal341) 
                    stream_21.add(char_literal20)
                    char_literal21=self.match(self.input, 22, self.FOLLOW_22_in_temporal343) 
                    stream_22.add(char_literal21)
                    low=self.match(self.input, INT, self.FOLLOW_INT_in_temporal347) 
                    stream_INT.add(low)
                    char_literal22=self.match(self.input, 24, self.FOLLOW_24_in_temporal349) 
                    stream_24.add(char_literal22)
                    high=self.match(self.input, INT, self.FOLLOW_INT_in_temporal353) 
                    stream_INT.add(high)
                    char_literal23=self.match(self.input, 23, self.FOLLOW_23_in_temporal355) 
                    stream_23.add(char_literal23)




                    # AST Rewrite
                    # elements: phi, high, low
                    # token labels: high, low
                    # rule labels: phi, retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 

                    retval.tree = root_0
                    stream_high = RewriteRuleTokenStream(self._adaptor, "token high", high)
                    stream_low = RewriteRuleTokenStream(self._adaptor, "token low", low)

                    if phi is not None:
                        stream_phi = RewriteRuleSubtreeStream(self._adaptor, "rule phi", phi.tree)
                    else:
                        stream_phi = RewriteRuleSubtreeStream(self._adaptor, "token phi", None)


                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 115:62: -> ^( WITHIN['W'] $phi $low $high)
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:115:65: ^( WITHIN['W'] $phi $low $high)
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(WITHIN, 'W'), root_1)

                    self._adaptor.addChild(root_1, stream_phi.nextTree())
                    self._adaptor.addChild(root_1, stream_low.nextNode())
                    self._adaptor.addChild(root_1, stream_high.nextNode())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0


                elif alt4 == 4:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:116:7: negation
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_negation_in_temporal380)
                    negation24 = self.negation()

                    self._state.following.pop()
                    self._adaptor.addChild(root_0, negation24.tree)


                retval.stop = self.input.LT(-1)


                retval.tree = self._adaptor.rulePostProcessing(root_0)
                self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:

            pass
        return retval

    # $ANTLR end "temporal"

    class negation_return(ParserRuleReturnScope):
        def __init__(self):
            super(twtlParser.negation_return, self).__init__()

            self.tree = None




    # $ANTLR start "negation"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:119:1: negation : ( NOT )? atom ;
    def negation(self, ):

        retval = self.negation_return()
        retval.start = self.input.LT(1)

        root_0 = None

        NOT25 = None
        atom26 = None


        NOT25_tree = None

        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:120:5: ( ( NOT )? atom )
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:120:7: ( NOT )? atom
                pass 
                root_0 = self._adaptor.nil()

                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:120:7: ( NOT )?
                alt5 = 2
                LA5_0 = self.input.LA(1)

                if (LA5_0 == NOT) :
                    alt5 = 1
                if alt5 == 1:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:120:8: NOT
                    pass 
                    NOT25=self.match(self.input, NOT, self.FOLLOW_NOT_in_negation399)

                    NOT25_tree = self._adaptor.createWithPayload(NOT25)
                    root_0 = self._adaptor.becomeRoot(NOT25_tree, root_0)




                self._state.following.append(self.FOLLOW_atom_in_negation404)
                atom26 = self.atom()

                self._state.following.pop()
                self._adaptor.addChild(root_0, atom26.tree)



                retval.stop = self.input.LT(-1)


                retval.tree = self._adaptor.rulePostProcessing(root_0)
                self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:

            pass
        return retval

    # $ANTLR end "negation"

    class atom_return(ParserRuleReturnScope):
        def __init__(self):
            super(twtlParser.atom_return, self).__init__()

            self.tree = None




    # $ANTLR start "atom"
    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:123:1: atom : ( TRUE | FALSE | PROP | '(' formula ')' );
    def atom(self, ):

        retval = self.atom_return()
        retval.start = self.input.LT(1)

        root_0 = None

        TRUE27 = None
        FALSE28 = None
        PROP29 = None
        char_literal30 = None
        char_literal32 = None
        formula31 = None


        TRUE27_tree = None
        FALSE28_tree = None
        PROP29_tree = None
        char_literal30_tree = None
        char_literal32_tree = None

        try:
            try:
                # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:124:5: ( TRUE | FALSE | PROP | '(' formula ')' )
                alt6 = 4
                LA6 = self.input.LA(1)
                if LA6 == TRUE:
                    alt6 = 1
                elif LA6 == FALSE:
                    alt6 = 2
                elif LA6 == PROP:
                    alt6 = 3
                elif LA6 == 25:
                    alt6 = 4
                else:
                    nvae = NoViableAltException("", 6, 0, self.input)

                    raise nvae

                if alt6 == 1:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:124:7: TRUE
                    pass 
                    root_0 = self._adaptor.nil()

                    TRUE27=self.match(self.input, TRUE, self.FOLLOW_TRUE_in_atom422)

                    TRUE27_tree = self._adaptor.createWithPayload(TRUE27)
                    self._adaptor.addChild(root_0, TRUE27_tree)



                elif alt6 == 2:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:125:7: FALSE
                    pass 
                    root_0 = self._adaptor.nil()

                    FALSE28=self.match(self.input, FALSE, self.FOLLOW_FALSE_in_atom431)

                    FALSE28_tree = self._adaptor.createWithPayload(FALSE28)
                    self._adaptor.addChild(root_0, FALSE28_tree)



                elif alt6 == 3:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:126:7: PROP
                    pass 
                    root_0 = self._adaptor.nil()

                    PROP29=self.match(self.input, PROP, self.FOLLOW_PROP_in_atom440)

                    PROP29_tree = self._adaptor.createWithPayload(PROP29)
                    self._adaptor.addChild(root_0, PROP29_tree)



                elif alt6 == 4:
                    # C:\\Users\\Cristian\\Dropbox\\work\\workspace\\TWTL\\src\\twtl.g:127:7: '(' formula ')'
                    pass 
                    root_0 = self._adaptor.nil()

                    char_literal30=self.match(self.input, 25, self.FOLLOW_25_in_atom449)
                    self._state.following.append(self.FOLLOW_formula_in_atom452)
                    formula31 = self.formula()

                    self._state.following.pop()
                    self._adaptor.addChild(root_0, formula31.tree)
                    char_literal32=self.match(self.input, 26, self.FOLLOW_26_in_atom454)


                retval.stop = self.input.LT(-1)


                retval.tree = self._adaptor.rulePostProcessing(root_0)
                self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:

            pass
        return retval

    # $ANTLR end "atom"


    # Delegated rules


 

    FOLLOW_formula_in_prog171 = frozenset([1])
    FOLLOW_disjunction_in_formula182 = frozenset([1])
    FOLLOW_conjunction_in_disjunction195 = frozenset([1, 5])
    FOLLOW_OR_in_disjunction198 = frozenset([6, 7, 11, 12, 13, 22, 25])
    FOLLOW_conjunction_in_disjunction201 = frozenset([1, 5])
    FOLLOW_concatenation_in_conjunction224 = frozenset([1, 4])
    FOLLOW_AND_in_conjunction227 = frozenset([6, 7, 11, 12, 13, 22, 25])
    FOLLOW_concatenation_in_conjunction230 = frozenset([1, 4])
    FOLLOW_temporal_in_concatenation252 = frozenset([1, 9])
    FOLLOW_CONCAT_in_concatenation255 = frozenset([6, 7, 11, 12, 13, 22, 25])
    FOLLOW_temporal_in_concatenation258 = frozenset([1, 9])
    FOLLOW_HOLD_in_temporal278 = frozenset([21])
    FOLLOW_21_in_temporal281 = frozenset([10])
    FOLLOW_INT_in_temporal286 = frozenset([11])
    FOLLOW_PROP_in_temporal288 = frozenset([1])
    FOLLOW_HOLD_in_temporal296 = frozenset([21])
    FOLLOW_21_in_temporal298 = frozenset([10])
    FOLLOW_INT_in_temporal302 = frozenset([6])
    FOLLOW_NOT_in_temporal304 = frozenset([11])
    FOLLOW_PROP_in_temporal308 = frozenset([1])
    FOLLOW_22_in_temporal333 = frozenset([6, 7, 11, 12, 13, 22, 25])
    FOLLOW_formula_in_temporal337 = frozenset([23])
    FOLLOW_23_in_temporal339 = frozenset([21])
    FOLLOW_21_in_temporal341 = frozenset([22])
    FOLLOW_22_in_temporal343 = frozenset([10])
    FOLLOW_INT_in_temporal347 = frozenset([24])
    FOLLOW_24_in_temporal349 = frozenset([10])
    FOLLOW_INT_in_temporal353 = frozenset([23])
    FOLLOW_23_in_temporal355 = frozenset([1])
    FOLLOW_negation_in_temporal380 = frozenset([1])
    FOLLOW_NOT_in_negation399 = frozenset([6, 7, 11, 12, 13, 22, 25])
    FOLLOW_atom_in_negation404 = frozenset([1])
    FOLLOW_TRUE_in_atom422 = frozenset([1])
    FOLLOW_FALSE_in_atom431 = frozenset([1])
    FOLLOW_PROP_in_atom440 = frozenset([1])
    FOLLOW_25_in_atom449 = frozenset([6, 7, 11, 12, 13, 22, 25])
    FOLLOW_formula_in_atom452 = frozenset([26])
    FOLLOW_26_in_atom454 = frozenset([1])



def main(argv, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr):
    from antlr3.main import ParserMain
    main = ParserMain("twtlLexer", twtlParser)
    main.stdin = stdin
    main.stdout = stdout
    main.stderr = stderr
    main.execute(argv)


if __name__ == '__main__':
    main(sys.argv)
