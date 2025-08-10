from pxr import Gf

MAX_ITEMS = 16

# A special quad-tree.
# init tree with a list of aabb or blank
# then add aabb success only if it not intersect with the items in tree already.
class QuadTree:
    def __init__(self, rect: Gf.Range2f, items):
        self._rect = rect
        self._center = rect.GetMidpoint()
        # Non-leaf
        self._nwse = []
        self.nw = self.ne = self.se = self.sw = None

        # the aabbs that in the leaf, will split when more than the MAX_ITEMS
        self._items = items

        if len(self._items) > MAX_ITEMS:
            self.split()

    def add_aabb(self, aabb: Gf.Range2f):
        lists = self._add_aabb_intenal(aabb)
        if lists == None:
            return False

        for list in lists:
            list.append(aabb)

        return True

    def _add_aabb_intenal(self, aabb: Gf.Range2f):
        if len(self._items) > MAX_ITEMS:
            self.split()

        if self.nw == None:
            return self._add_in_leaf(aabb)
        else:
            return self._add_in_nonleaf(aabb)

    def _add_in_leaf(self, aabb: Gf.Range2f):
        for item in self._items:
            if not Gf.Range2f.GetIntersection(item, aabb).IsEmpty():
                return None
        return [self._items]

    def _add_in_nonleaf(self, aabb: Gf.Range2f):
        for item in self._nwse:
            if not Gf.Range2f.GetIntersection(item, aabb).IsEmpty():
                return None

        in_nw = aabb.GetMin()[0] <= self._center[0] and aabb.GetMax()[1] >= self._center[1]
        in_sw = aabb.GetMin()[0] <= self._center[0] and aabb.GetMin()[1] <= self._center[1]
        in_ne = aabb.GetMax()[0] >= self._center[0] and aabb.GetMax()[1] >= self._center[1]
        in_se = aabb.GetMax()[0] >= self._center[0] and aabb.GetMin()[1] <= self._center[1]

        lists = []
        if in_nw:
            result = self.nw._add_aabb_intenal(aabb)
            if result:
                lists.extend(result)
            else:
                return None

        if in_ne:
            result = self.ne._add_aabb_intenal(aabb)
            if result:
                lists.extend(result)
            else:
                return None

        if in_sw:
            result = self.sw._add_aabb_intenal(aabb)
            if result:
                lists.extend(result)
            else:
                return None

        if in_se:
            result = self.se._add_aabb_intenal(aabb)
            if result:
                lists.extend(result)
            else:
                return None

        if in_nw and in_ne and in_se and in_sw:
            return [self._nwse]

        return lists

    def split(self):
        nw_items = []
        ne_items = []
        sw_items = []
        se_items = []

        for item in self._items:
            in_nw = item.GetMin()[0] <= self._center[0] and item.GetMax()[1] >= self._center[1]
            in_sw = item.GetMin()[0] <= self._center[0] and item.GetMin()[1] <= self._center[1]
            in_ne = item.GetMax()[0] >= self._center[0] and item.GetMax()[1] >= self._center[1]
            in_se = item.GetMax()[0] >= self._center[0] and item.GetMin()[1] <= self._center[1]

            if in_nw and in_ne and in_se and in_sw:
                self._nwse.append(item)
                continue

            if in_nw:
                nw_items.append(item)
            if in_ne:
                ne_items.append(item)
            if in_sw:
                sw_items.append(item)
            if in_se:
                se_items.append(item)

        self.sw = QuadTree(self._rect.GetQuadrant(0), sw_items)
        self.se = QuadTree(self._rect.GetQuadrant(1), se_items)
        self.nw = QuadTree(self._rect.GetQuadrant(2), nw_items)
        self.ne = QuadTree(self._rect.GetQuadrant(3), ne_items)

        self._items = []

    def contains(self, aabb: Gf.Range2f):
        return self._rect.Contains(aabb)

    # traversing tree, print debug infomation
    # def print_info(self):
    #     print("----")
    #     print(f" rect {self._center}, {self._rect}")
    #     print(f"nswe {self._nwse}")
    #     print(f"items {self._items}")

    #     if self.nw:
    #         print("nw")
    #         self.nw.print_info()
    #         print("ne")
    #         self.ne.print_info()
    #         print("sw")
    #         self.sw.print_info()
    #         print("se")
    #         self.se.print_info()
