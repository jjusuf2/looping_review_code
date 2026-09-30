function [eventTimesI, EventTimesTotal, meanEventTimePerTrack] = DetectEventTimes(bPost)
[L, N] = size(bPost);

EventTimesTotal = [];
for l=1:L
    eventTimes = -1*ones(1, N);
    count = 1;
    for k=1:N

        if (k>1)
            if (bPost(l, k)==1 && bPost(l,k-1)==1)
                eventTimes(count) = eventTimes(count) + 1;
            elseif (bPost(l,k)==1 && bPost(l,k-1)==0)
                count = count + 1;
                eventTimes(count) = 1;
            end
        else
            if (bPost(l,k)==1)
                eventTimes(count) = 1;
                count = 1;
            else
                count = 0;
            end
        end
    end

    idx = find(eventTimes==-1);
    eventTimesI{l} = eventTimes(1:idx(1)-1);
    meanEventTimePerTrack(l) = mean(eventTimes(1:idx(1)-1));

    EventTimesTotal = [EventTimesTotal, eventTimesI{l}];
end