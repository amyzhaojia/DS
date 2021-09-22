import copy
import numpy as np

# twoSum
def twoSum(nums, target):
    for i in range(len(nums)):
        a = nums[i]
        b = target - a
        num = copy.deepcopy(nums)
        del num[i]
        if b in num:
            break
    m = i
    if b == a:
        c = int(np.where(np.array(nums) == b)[0][1])
    else:
        c = int(np.where(np.array(nums) == b)[0][0])
    return [i, c]

def twoSum1(nums, target):
    for i in range(len(nums) - 1):
        base = nums[i]
        other = target - base
        if other in nums[i + 1:]:
            # 这里注意index设置start，避免出现target = 6，[3,3]返回[0,0]的错误
            return [i, nums.index(other, i + 1)]

def twoSum_hash(nums, target):
    tmp = {}
    for k, v in enumerate(nums):
        if target - v in tmp:
            return [tmp[target - v], k]
        tmp[v] = k


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def addTwoNumbers(self, l1, l2):

    #1.还原两个非负整数；
    num1 = str(l1.val)
    while l1.next:
        l1 = l1.next
        num1 += str(l1.val)
    num1 = int(num1[::-1])
    num2 = str(l2.val)
    while l2.next:
        l2 = l2.next
        num2 += str(l2.val)
    num2 = int(num2[::-1])
    num = str(num1 + num2)
    L = len(num)
    result = ListNode(int(num[0]))
    for i in range(1, L):
        result = ListNode(int(num[i]), result)
    return result

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        num1 = 0
        num2 = 0
        i = 0
        j = 0
        num = ListNode()
        while l1 is not None:
            num1 = l1.val*10**i+num1
            l1 = l1.next
            i +=1
        while l2 is not None:
            num2 = l2.val*10**j+num2
            l2 = l2.next
            j +=1
        num.val = num1+num2
        l3 = ListNode(val=int(str(num.val)[-1]))
        number = l3
        for k in range(len(str(num.val))-1):
            number.next = ListNode(val=int(str(num.val)[-k-2]))
            number = number.next
        return l3



def lengthOfLongestSubstring(s):
    st = {}
    i, ans = 0, 0
    for j in range(len(s)):
        if s[j] in st:
            i = max(st[s[j]], i)
        ans = max(ans, j - i + 1)
        st[s[j]] = j + 1
    return ans


def lengthOfLongestSubstring_1(s):
    long_s = ''
    cal_s = ''
    cal_s1 = ''
    s_dic = {}
    left = 0
    for i in range(len(s)):
        if s[i] in cal_s:
            left = max(s_dic[s[i]],left)
            cal_s1 = s[left+1:i+1]
        else:
            cal_s1 = cal_s1 + s[i]
            cal_s = cal_s + s[i]
        if len(cal_s1)>len(long_s):
            long_s = cal_s1
        s_dic[s[i]] = i
    return len(long_s) #,long_s


if __name__ == '__main__':
    # nums = [3,3]
    # target = 6
    # [m,n] = twoSum_hash(nums,target)
    s = 'pwwkew'
    m = lengthOfLongestSubstring_1(s)
    print(m)