"""
  @author: yigit.yildirim@boun.edu.tr
"""
import numpy as np
import sys
from env import IRLMDP


class RLAgent:
    def __init__(self):
        self.vi_loop = 2000
        self.env = IRLMDP()
        self.policy = np.zeros((len(self.env.states), len(self.env.actions)))

    def appr_vi(self, rewards):
        v = np.ones((len(self.env.states), 1)) * -sys.float_info.max
        q = np.zeros((len(self.env.states), len(self.env.actions)))

        for i in range(self.vi_loop - 1):
            v[self.env.goal_id] = 0
            for s in range(len(self.env.states)):
                q[s, :] = np.matmul(self.env.transition[s, :, :], v).T + rewards[s]

            # v = softmax_a q
            # one problem:
            # when np.sum(np.exp(q), axis=1) = 0, division by 0. In this case v = 0
            expq = np.exp(q)
            sumexpq = np.sum(expq, axis=1)
            nonzero_ids = np.where(sumexpq != 0)
            zero_ids = np.where(sumexpq == 0)
            v[nonzero_ids, 0] = np.exp(np.max(q[nonzero_ids], axis=1)) / sumexpq[nonzero_ids]
            v[zero_ids, 0] = -sys.float_info.max

            print('\rVI Loop: {}'.format((i + 1)), end='')

        v[self.env.goal_id] = 0
        # current MaxEnt policy:
        advantage = q - np.reshape(v, (len(self.env.states), 1))
        self.policy = np.exp(advantage)


if __name__ == "__main__":
    ra = RLAgent()
    ra.appr_vi(np.asarray([[-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.61006389], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.6100639], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006391], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006392], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006393], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006394], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006395], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006396], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006397], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006398], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.61006399], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.610064], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006401], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006402], [-745.61006402], [-745.61006402], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006403], [-745.61006404], [-745.61006404], [-745.61006404], [-745.61006404], [-745.61006404], [-745.61006404], [-745.61006404], [-745.61006404], [-745.61006404], [-745.61006404], [-745.61006405], [-745.61006405], [-745.61006405], [-745.61006405], [-745.61006405], [-745.61006405], [-745.61006405], [-745.61006405], [-745.61006405], [-745.61006405], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006405], [-745.61006405], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006406], [-745.61006407], [-745.61006407], [-745.61006407], [-745.61006407], [-745.61006407], [-745.61006407], [-745.61006407], [-745.61006407], [-745.61006408], [-745.61006408], [-745.61006408], [-745.61006408], [-745.61006408], [-745.61006408], [-745.61006408], [-745.61006408], [-745.61006409], [-745.61006409], [-745.61006409], [-745.61006409], [-745.61006409], [-745.61006409], [-745.61006409], [-745.6100641], [-745.6100641], [-745.6100641], [-745.6100641], [-745.6100641], [-745.6100641], [-745.61006409], [-745.61006409], [-745.61006409], [-745.6100641], [-745.6100641], [-745.6100641], [-745.6100641], [-745.6100641], [-745.6100641], [-745.6100641], [-745.61006411], [-745.61006411], [-745.61006411], [-745.61006411], [-745.61006411], [-745.61006411], [-745.61006411], [-745.61006412], [-745.61006412], [-745.61006412], [-745.61006412], [-745.61006412], [-745.61006412], [-745.61006412], [-745.61006413], [-745.61006413], [-745.61006413], [-745.61006413], [-745.61006413], [-745.61006413], [-745.61006414], [-745.61006414], [-745.61006414], [-745.61006414], [-745.61006414], [-745.61006414], [-745.61006415], [-745.61006415], [-745.61006415], [-745.61006415], [-745.61006414], [-745.61006414], [-745.61006414], [-745.61006414], [-745.61006414], [-745.61006415], [-745.61006415], [-745.61006415], [-745.61006415], [-745.61006415], [-745.61006415], [-745.61006416], [-745.61006416], [-745.61006416], [-745.61006416], [-745.61006416], [-745.61006417], [-745.61006417], [-745.61006417], [-745.61006417], [-745.61006417], [-745.61006417], [-745.61006418], [-745.61006418], [-745.61006418], [-745.61006418], [-745.61006418], [-745.61006419], [-745.61006419], [-745.61006419], [-745.61006419], [-745.61006419], [-745.6100642], [-745.6100642], [-745.6100642], [-745.6100642], [-745.61006421], [-745.61006421], [-745.61006421], [-745.61006421], [-745.61006419], [-745.6100642], [-745.6100642], [-745.6100642], [-745.6100642], [-745.6100642], [-745.61006421], [-745.61006421], [-745.61006421], [-745.61006421], [-745.61006421], [-745.61006422], [-745.61006422], [-745.61006422], [-745.61006422], [-745.61006423], [-745.61006423], [-745.61006423], [-745.61006423], [-745.61006423], [-745.61006424], [-745.61006424], [-745.61006424], [-745.61006424], [-745.61006425], [-745.61006425], [-745.61006425], [-745.61006425], [-745.61006426], [-745.61006426], [-745.61006426], [-745.61006426], [-745.61006427], [-745.61006427], [-745.61006427], [-745.61006427], [-745.61006428], [-745.61006428], [-745.61006428], [-745.61006429], [-745.61006426], [-745.61006426], [-745.61006427], [-745.61006427], [-745.61006427], [-745.61006427], [-745.61006428], [-745.61006428], [-745.61006428], [-745.61006429], [-745.61006429], [-745.61006429], [-745.61006429], [-745.6100643], [-745.6100643], [-745.6100643], [-745.6100643], [-745.61006431], [-745.61006431], [-745.61006431], [-745.61006432], [-745.61006432], [-745.61006432], [-745.61006433], [-745.61006433], [-745.61006433], [-745.61006433], [-745.61006434], [-745.61006434], [-745.61006434], [-745.61006435], [-745.61006435], [-745.61006435], [-745.61006436], [-745.61006436], [-745.61006436], [-745.61006437], [-745.61006437], [-745.61006437], [-745.61006438], [-745.61006435], [-745.61006435], [-745.61006435], [-745.61006436], [-745.61006436], [-745.61006436], [-745.61006437], [-745.61006437], [-745.61006437], [-745.61006437], [-745.61006438], [-745.61006438], [-745.61006439], [-745.61006439], [-745.61006439], [-745.6100644], [-745.6100644], [-745.6100644], [-745.61006441], [-745.61006441], [-745.61006441], [-745.61006442], [-745.61006442], [-745.61006442], [-745.61006443], [-745.61006443], [-745.61006444], [-745.61006444], [-745.61006444], [-745.61006445], [-745.61006445], [-745.61006445], [-745.61006446], [-745.61006446], [-745.61006447], [-745.61006447], [-745.61006448], [-745.61006448], [-745.61006448], [-745.61006449], [-745.61006445], [-745.61006445], [-745.61006446], [-745.61006446], [-745.61006446], [-745.61006447], [-745.61006447], [-745.61006448], [-745.61006448], [-745.61006448], [-745.61006449], [-745.61006449], [-745.6100645], [-745.6100645], [-745.61006451], [-745.61006451], [-745.61006451], [-745.61006452], [-745.61006452], [-745.61006453], [-745.61006453], [-745.61006454], [-745.61006454], [-745.61006455], [-745.61006455], [-745.61006456], [-745.61006456], [-745.61006456], [-745.61006457], [-745.61006457], [-745.61006458], [-745.61006458], [-745.61006459], [-745.61006459], [-745.6100646], [-745.6100646], [-745.61006461], [-745.61006461], [-745.61006462], [-745.61006462], [-745.61006458], [-745.61006458], [-745.61006458], [-745.61006459], [-745.61006459], [-745.6100646], [-745.6100646], [-745.61006461], [-745.61006461], [-745.61006462], [-745.61006462], [-745.61006463], [-745.61006463], [-745.61006464], [-745.61006465], [-745.61006465], [-745.61006466], [-745.61006466], [-745.61006467], [-745.61006467], [-745.61006468], [-745.61006468], [-745.61006469], [-745.61006469], [-745.6100647], [-745.61006471], [-745.61006471], [-745.61006472], [-745.61006472], [-745.61006473], [-745.61006474], [-745.61006474], [-745.61006475], [-745.61006475], [-745.61006476], [-745.61006477], [-745.61006477], [-745.61006478], [-745.61006479], [-745.61006479], [-745.61006473], [-745.61006474], [-745.61006474], [-745.61006475], [-745.61006475], [-745.61006476], [-745.61006477], [-745.61006477], [-745.61006478], [-745.61006478], [-745.61006479], [-745.6100648], [-745.6100648], [-745.61006481], [-745.61006482], [-745.61006482], [-745.61006483], [-745.61006484], [-745.61006484], [-745.61006485], [-745.61006486], [-745.61006486], [-745.61006487], [-745.61006488], [-745.61006488], [-745.61006489], [-745.6100649], [-745.61006491], [-745.61006491], [-745.61006492], [-745.61006493], [-745.61006494], [-745.61006494], [-745.61006495], [-745.61006496], [-745.61006497], [-745.61006497], [-745.61006498], [-745.61006499], [-745.610065], [-745.61006492], [-745.61006493], [-745.61006493], [-745.61006494], [-745.61006495], [-745.61006496], [-745.61006496], [-745.61006497], [-745.61006498], [-745.61006499], [-745.61006499], [-745.610065], [-745.61006501], [-745.61006502], [-745.61006503], [-745.61006504], [-745.61006504], [-745.61006505], [-745.61006506], [-745.61006507], [-745.61006508], [-745.61006509], [-745.61006509], [-745.6100651], [-745.61006511], [-745.61006512], [-745.61006513], [-745.61006514], [-745.61006515], [-745.61006516], [-745.61006517], [-745.61006517], [-745.61006518], [-745.61006519], [-745.6100652], [-745.61006521], [-745.61006522], [-745.61006523], [-745.61006524], [-745.61006525], [-745.61006515], [-745.61006516], [-745.61006517], [-745.61006518], [-745.61006519], [-745.6100652], [-745.61006521], [-745.61006522], [-745.61006523], [-745.61006524], [-745.61006525], [-745.61006525], [-745.61006526], [-745.61006527], [-745.61006528], [-745.61006529], [-745.6100653], [-745.61006532], [-745.61006533], [-745.61006534], [-745.61006535], [-745.61006536], [-745.61006537], [-745.61006538], [-745.61006539], [-745.6100654], [-745.61006541], [-745.61006542], [-745.61006543], [-745.61006544], [-745.61006546], [-745.61006547], [-745.61006548], [-745.61006549], [-745.6100655], [-745.61006551], [-745.61006553], [-745.61006554], [-745.61006555], [-745.61006556], [-745.61006544], [-745.61006545], [-745.61006546], [-745.61006547], [-745.61006548], [-745.61006549], [-745.6100655], [-745.61006552], [-745.61006553], [-745.61006554], [-745.61006555], [-745.61006556], [-745.61006558], [-745.61006559], [-745.6100656], [-745.61006561], [-745.61006563], [-745.61006564], [-745.61006565], [-745.61006566], [-745.61006568], [-745.61006569], [-745.6100657], [-745.61006572], [-745.61006573], [-745.61006574], [-745.61006576], [-745.61006577], [-745.61006578], [-745.6100658], [-745.61006581], [-745.61006583], [-745.61006584], [-745.61006586], [-745.61006587], [-745.61006588], [-745.6100659], [-745.61006591], [-745.61006593], [-745.61006594], [-745.61006579], [-745.6100658], [-745.61006581], [-745.61006583], [-745.61006584], [-745.61006586], [-745.61006587], [-745.61006588], [-745.6100659], [-745.61006591], [-745.61006593], [-745.61006594], [-745.61006596], [-745.61006597], [-745.61006599], [-745.610066], [-745.61006602], [-745.61006604], [-745.61006605], [-745.61006607], [-745.61006608], [-745.6100661], [-745.61006612], [-745.61006613], [-745.61006615], [-745.61006617], [-745.61006618], [-745.6100662], [-745.61006622], [-745.61006623], [-745.61006625], [-745.61006627], [-745.61006629], [-745.6100663], [-745.61006632], [-745.61006634], [-745.61006636], [-745.61006638], [-745.61006639], [-745.61006641], [-745.61006621], [-745.61006623], [-745.61006625], [-745.61006627], [-745.61006628], [-745.6100663], [-745.61006632], [-745.61006634], [-745.61006635], [-745.61006637], [-745.61006639], [-745.61006641], [-745.61006643], [-745.61006645], [-745.61006647], [-745.61006649], [-745.6100665], [-745.61006652], [-745.61006654], [-745.61006656], [-745.61006658], [-745.6100666], [-745.61006662], [-745.61006664], [-745.61006666], [-745.61006668], [-745.6100667], [-745.61006673], [-745.61006675], [-745.61006677], [-745.61006679], [-745.61006681], [-745.61006683], [-745.61006685], [-745.61006688], [-745.6100669], [-745.61006692], [-745.61006694], [-745.61006697], [-745.61006699], [-745.61006674], [-745.61006676], [-745.61006678], [-745.6100668], [-745.61006682], [-745.61006685], [-745.61006687], [-745.61006689], [-745.61006691], [-745.61006694], [-745.61006696], [-745.61006698], [-745.610067], [-745.61006703], [-745.61006705], [-745.61006707], [-745.6100671], [-745.61006712], [-745.61006715], [-745.61006717], [-745.6100672], [-745.61006722], [-745.61006725], [-745.61006727], [-745.6100673], [-745.61006732], [-745.61006735], [-745.61006737], [-745.6100674], [-745.61006743], [-745.61006745], [-745.61006748], [-745.61006751], [-745.61006753], [-745.61006756], [-745.61006759], [-745.61006761], [-745.61006764], [-745.61006767], [-745.6100677], [-745.61006738], [-745.61006741], [-745.61006744], [-745.61006746], [-745.61006749], [-745.61006752], [-745.61006754], [-745.61006757], [-745.6100676], [-745.61006763], [-745.61006766], [-745.61006768], [-745.61006771], [-745.61006774], [-745.61006777], [-745.6100678], [-745.61006783], [-745.61006786], [-745.61006789], [-745.61006792], [-745.61006795], [-745.61006798], [-745.61006801], [-745.61006804], [-745.61006807], [-745.6100681], [-745.61006814], [-745.61006817], [-745.6100682], [-745.61006823], [-745.61006827], [-745.6100683], [-745.61006833], [-745.61006837], [-745.6100684], [-745.61006843], [-745.61006847], [-745.6100685], [-745.61006854], [-745.61006857], [-745.61006817], [-745.61006821], [-745.61006824], [-745.61006827], [-745.61006831], [-745.61006834], [-745.61006837], [-745.61006841], [-745.61006844], [-745.61006848], [-745.61006851], [-745.61006855], [-745.61006858], [-745.61006862], [-745.61006865], [-745.61006869], [-745.61006873], [-745.61006876], [-745.6100688], [-745.61006884], [-745.61006887], [-745.61006891], [-745.61006895], [-745.61006899], [-745.61006903], [-745.61006907], [-745.61006911], [-745.61006915], [-745.61006919], [-745.61006923], [-745.61006927], [-745.61006931], [-745.61006935], [-745.61006939], [-745.61006943], [-745.61006947], [-745.61006952], [-745.61006956], [-745.6100696], [-745.61006965], [-745.61006915], [-745.61006919], [-745.61006923], [-745.61006927], [-745.61006931], [-745.61006935], [-745.61006939], [-745.61006943], [-745.61006948], [-745.61006952], [-745.61006956], [-745.61006961], [-745.61006965], [-745.61006969], [-745.61006974], [-745.61006978], [-745.61006983], [-745.61006987], [-745.61006992], [-745.61006997], [-745.61007001], [-745.61007006], [-745.61007011], [-745.61007015], [-745.6100702], [-745.61007025], [-745.6100703], [-745.61007035], [-745.6100704], [-745.61007045], [-745.6100705], [-745.61007055], [-745.6100706], [-745.61007065], [-745.6100707], [-745.61007075], [-745.61007081], [-745.61007086], [-745.61007091], [-745.61007097], [-745.61007034], [-745.61007039], [-745.61007044], [-745.61007049], [-745.61007054], [-745.61007059], [-745.61007064], [-745.6100707], [-745.61007075], [-745.6100708], [-745.61007085], [-745.61007091], [-745.61007096], [-745.61007102], [-745.61007107], [-745.61007113], [-745.61007118], [-745.61007124], [-745.6100713], [-745.61007135], [-745.61007141], [-745.61007147], [-745.61007153], [-745.61007159], [-745.61007165], [-745.61007171], [-745.61007177], [-745.61007183], [-745.61007189], [-745.61007195], [-745.61007201], [-745.61007207], [-745.61007214], [-745.6100722], [-745.61007226], [-745.61007233], [-745.61007239], [-745.61007246], [-745.61007253], [-745.61007259], [-745.6100718], [-745.61007186], [-745.61007193], [-745.61007199], [-745.61007205], [-745.61007212], [-745.61007218], [-745.61007225], [-745.61007231], [-745.61007238], [-745.61007244], [-745.61007251], [-745.61007258], [-745.61007264], [-745.61007271], [-745.61007278], [-745.61007285], [-745.61007292], [-745.61007299], [-745.61007306], [-745.61007313], [-745.6100732], [-745.61007328], [-745.61007335], [-745.61007342], [-745.6100735], [-745.61007357], [-745.61007365], [-745.61007372], [-745.6100738], [-745.61007387], [-745.61007395], [-745.61007403], [-745.61007411], [-745.61007419], [-745.61007427], [-745.61007435], [-745.61007443], [-745.61007451], [-745.61007459], [-745.6100736], [-745.61007368], [-745.61007376], [-745.61007383], [-745.61007391], [-745.61007399], [-745.61007407], [-745.61007415], [-745.61007423], [-745.61007431], [-745.61007439], [-745.61007448], [-745.61007456], [-745.61007464], [-745.61007473], [-745.61007481], [-745.6100749], [-745.61007498], [-745.61007507], [-745.61007516], [-745.61007525], [-745.61007534], [-745.61007543], [-745.61007552], [-745.61007561], [-745.6100757], [-745.61007579], [-745.61007588], [-745.61007598], [-745.61007607], [-745.61007617], [-745.61007626], [-745.61007636], [-745.61007646], [-745.61007656], [-745.61007665], [-745.61007675], [-745.61007685], [-745.61007696], [-745.61007706], [-745.61007581], [-745.61007591], [-745.610076], [-745.6100761], [-745.6100762], [-745.61007629], [-745.61007639], [-745.61007649], [-745.61007659], [-745.61007669], [-745.61007679], [-745.6100769], [-745.610077], [-745.6100771], [-745.61007721], [-745.61007731], [-745.61007742], [-745.61007753], [-745.61007763], [-745.61007774], [-745.61007785], [-745.61007796], [-745.61007807], [-745.61007818], [-745.6100783], [-745.61007841], [-745.61007852], [-745.61007864], [-745.61007876], [-745.61007887], [-745.61007899], [-745.61007911], [-745.61007923], [-745.61007935], [-745.61007947], [-745.61007959], [-745.61007972], [-745.61007984], [-745.61007997], [-745.61008009], [-745.61007853], [-745.61007865], [-745.61007876], [-745.61007888], [-745.610079], [-745.61007913], [-745.61007925], [-745.61007937], [-745.61007949], [-745.61007962], [-745.61007975], [-745.61007987], [-745.61008], [-745.61008013], [-745.61008026], [-745.61008039], [-745.61008052], [-745.61008065], [-745.61008079], [-745.61008092], [-745.61008106], [-745.61008119], [-745.61008133], [-745.61008147], [-745.61008161], [-745.61008175], [-745.61008189], [-745.61008203], [-745.61008218], [-745.61008232], [-745.61008247], [-745.61008261], [-745.61008276], [-745.61008291], [-745.61008306], [-745.61008321], [-745.61008336], [-745.61008352], [-745.61008367], [-745.61008383], [-745.61008187], [-745.61008201], [-745.61008216], [-745.61008231], [-745.61008246], [-745.61008261], [-745.61008276], [-745.61008291], [-745.61008307], [-745.61008322], [-745.61008338], [-745.61008353], [-745.61008369], [-745.61008385], [-745.61008401], [-745.61008417], [-745.61008434], [-745.6100845], [-745.61008467], [-745.61008483], [-745.610085], [-745.61008517], [-745.61008534], [-745.61008551], [-745.61008568], [-745.61008586], [-745.61008603], [-745.61008621], [-745.61008639], [-745.61008657], [-745.61008675], [-745.61008693], [-745.61008711], [-745.6100873], [-745.61008748], [-745.61008767], [-745.61008786], [-745.61008805], [-745.61008824], [-745.61008843], [-745.61008598], [-745.61008616], [-745.61008634], [-745.61008652], [-745.61008671], [-745.61008689], [-745.61008708], [-745.61008727], [-745.61008746], [-745.61008765], [-745.61008784], [-745.61008804], [-745.61008824], [-745.61008843], [-745.61008863], [-745.61008883], [-745.61008903], [-745.61008924], [-745.61008944], [-745.61008965], [-745.61008986], [-745.61009006], [-745.61009028], [-745.61009049], [-745.6100907], [-745.61009092], [-745.61009113], [-745.61009135], [-745.61009157], [-745.6100918], [-745.61009202], [-745.61009225], [-745.61009247], [-745.6100927], [-745.61009293], [-745.61009316], [-745.6100934], [-745.61009363], [-745.61009387], [-745.61009411]]))
    print('f')