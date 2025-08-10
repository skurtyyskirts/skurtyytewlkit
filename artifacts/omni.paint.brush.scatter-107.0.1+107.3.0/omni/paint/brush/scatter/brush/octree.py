from pxr import Gf

MAX_ITEMS = 16


# A special oc-tree.
# init tree with a list of aabb or blank
# then add aabb success only if it not intersect with the exist items in tree.
class OcTree:
    def __init__(self, rect: Gf.Range3d, items):
        self._rect = rect
        self._center = rect.GetMidpoint()
        self._children = None
        self._center_items = []

        # the aabbs that in the leaf, will split when more than the MAX_ITEMS
        self._items = items

        if len(self._items) > MAX_ITEMS:
            self.split()

    def add_aabb(self, aabb: Gf.Range3d):
        lists = self._add_aabb_intenal(aabb)
        if not lists:
            # print(f"add false {aabb} ")
            return False

        # print(f"add true  {aabb} ")
        for list in lists:
            list.append(aabb)

        return True

    def _add_aabb_intenal(self, aabb: Gf.Range3d):
        if len(self._items) > MAX_ITEMS:
            self.split()

        if self._children:
            return self._add_in_children(aabb)
        else:
            return self._add_in_self(aabb)

    def _add_in_self(self, aabb: Gf.Range3d):
        for item in self._items:
            if not Gf.Range3d.GetIntersection(item, aabb).IsEmpty():
                return None
        return [self._items]

    def _add_in_children(self, aabb: Gf.Range3d):
        for item in self._center_items:
            if not Gf.Range3d.GetIntersection(item, aabb).IsEmpty():
                return None

        lists = []
        in_children = 0
        for i in range(8):
            if not Gf.Range3d.GetIntersection(aabb, self._rect.GetOctant(i)).IsEmpty():
                in_children += 1
                result = self._children[i]._add_aabb_intenal(aabb)
                if result:
                    lists.extend(result)
                else:
                    return None

        if in_children == 8:
            return [self._center_items]
        else:
            return lists

    def split(self):
        children_items = [[] for i in range(8)]

        for item in self._items:
            in_0 = not Gf.Range3d.GetIntersection(item, self._rect.GetOctant(0)).IsEmpty()
            in_1 = not Gf.Range3d.GetIntersection(item, self._rect.GetOctant(1)).IsEmpty()
            in_2 = not Gf.Range3d.GetIntersection(item, self._rect.GetOctant(2)).IsEmpty()
            in_3 = not Gf.Range3d.GetIntersection(item, self._rect.GetOctant(3)).IsEmpty()
            in_4 = not Gf.Range3d.GetIntersection(item, self._rect.GetOctant(4)).IsEmpty()
            in_5 = not Gf.Range3d.GetIntersection(item, self._rect.GetOctant(5)).IsEmpty()
            in_6 = not Gf.Range3d.GetIntersection(item, self._rect.GetOctant(6)).IsEmpty()
            in_7 = not Gf.Range3d.GetIntersection(item, self._rect.GetOctant(7)).IsEmpty()

            if in_0 and in_1 and in_2 and in_3 and in_4 and in_5 and in_6 and in_7:
                self._center_items.append(item)
                continue

            if in_0:
                children_items[0].append(item)
            if in_1:
                children_items[1].append(item)
            if in_2:
                children_items[2].append(item)
            if in_3:
                children_items[3].append(item)
            if in_4:
                children_items[4].append(item)
            if in_5:
                children_items[5].append(item)
            if in_6:
                children_items[6].append(item)
            if in_7:
                children_items[7].append(item)

        self._children = []
        for i in range(8):
            self._children.append(OcTree(self._rect.GetOctant(i), children_items[i]))

        self._items.clear()

    def contains(self, aabb: Gf.Range3d):
        return self._rect.Contains(aabb)
