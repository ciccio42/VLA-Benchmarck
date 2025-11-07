from multi_task_robosuite_env.arena import TableArena
from robosuite.utils.mjcf_utils import array_to_string

class PegsArena(TableArena):
    """
    Workspace that contains a tabletop with two fixed pegs.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
    """

    def __init__(
        self,
        table_full_size=(0.45, 0.69, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0),
        peg_positions={'peg1': [0.23, 0.2, 0.85], 'peg2': [0.23, 0.0, 0.85], 'peg3': [0.23, -0.2, 0.85]}
    ):
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            xml="pegs_arena.xml"
        )

        # Get references to peg bodies
        self.peg1_body = self.worldbody.find("./body[@name='peg1']")
        self.peg1_body.set("pos",array_to_string(peg_positions['peg1']))

        self.peg2_body = self.worldbody.find("./body[@name='peg2']")
        self.peg2_body.set("pos",array_to_string(peg_positions['peg2']))
        
        self.peg3_body = self.worldbody.find("./body[@name='peg3']")
        self.peg3_body.set("pos",array_to_string(peg_positions['peg3']))
